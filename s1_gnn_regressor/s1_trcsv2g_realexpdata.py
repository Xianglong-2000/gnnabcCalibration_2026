import numpy as np
import pandas as pd
import os
import sys
import shutil
import pickle as pk
import networkx as nx
from pathlib import Path
import subprocess
import torch
from torch_geometric.data import Data


def flatten_node_attributes(node_attr):
        """Flatten the node attributes to ensure all values are numeric."""
        flat_attrs = []
        for attr in node_attr.values():
            if isinstance(attr, (tuple, list)):
                flat_attrs.extend(attr)
            else:
                flat_attrs.append(attr)
        return flat_attrs


def generate_parent_df(df):
    # ImageNumber = stepNum
    parent_df = df[["ImageNumber", "ObjectNumber", "TrackObjects_ParentImageNumber_50", "TrackObjects_ParentObjectNumber_50"]].copy(deep=True)
    parent_df['Relationship'] = 'Parent'
    parent_df.columns = ["First Image Number", "First Object Number", "Second Image Number", "Second Object Number", "Relationship"]
    parent_df = parent_df[["Relationship", "First Image Number", "First Object Number", "Second Image Number", "Second Object Number"]].copy(deep=True)
    return parent_df


def smart_cast(val):
    try:
        return float(val)
    except ValueError:
        if val == 'True':
            return float(1)
        elif val == 'False':
            return float(0)
        elif val == 'TRUE':
            return float(1)
        elif val == 'FALSE':
            return float(0)
        elif val == 'YFP':
            return float(1)
        else:
            return val  # keep as string if not a number


def load_exp_graph_data(graph_ID:str, node_features_file:str, neighbours_edge_file:str, include_columns:list):
    
    # Create the y vector, y = [gamma, reg_param]
    print("the current real-exp sample is ", graph_ID)
    
    # Read though the csv files and create the df
    node_features_df = pd.read_csv(node_features_file, dtype=str)
    neighbours_edge_df = pd.read_csv(neighbours_edge_file, dtype=str)

    parents_edge_df = generate_parent_df(node_features_df)
    edge_features_df = pd.concat([parents_edge_df, neighbours_edge_df])

    for col in node_features_df.columns:
        node_features_df[col] = node_features_df[col].map(smart_cast)

    for col in edge_features_df.columns:
        edge_features_df[col] = edge_features_df[col].map(smart_cast)

    node_features_df['dir'] = node_features_df['dir'].apply(eval).apply(lambda x: tuple(map(float, x)))
    vec_df = pd.DataFrame(node_features_df['dir'].tolist(), columns=['dir_1', 'dir_2'])
    node_features_df = pd.concat([node_features_df.drop(columns=['dir']), vec_df], axis=1)

    # print(len(include_columns))
    node_features_df = node_features_df[include_columns]
    node_features_df = node_features_df.fillna(node_features_df.mean(numeric_only=True))

    node_index_df = node_features_df.iloc[:, :4]
    node_features_df = node_features_df.iloc[:, 4:]

    df_normalized = node_features_df.copy()

    # Standard normalization
    binary_columns = ["divideFlag", "Unexpected_End", "Unexpected_Beginning"]
    for col in node_features_df.columns:
        if col not in binary_columns:
            df_normalized[col] = (node_features_df[col] - mean_dict[col])/std_dict[col]
            ##print(mean_dict[col], std_dict[col])
    ##print(df_normalized["divideFlag"])
    ##print(df_normalized.head())

    node_features_df = df_normalized

    # print(node_features_df.dtypes)
    # print(node_features_df.head(3))
    # print(len(node_features_df.columns))
    # print(node_features_df.corr().to_string())

    # Create the graph
    G = nx.MultiDiGraph()

    # Add nodes with their features as attributes and label them by their index
    for idx, row in node_features_df.iterrows():
        node_id = idx  # Use the index as a unique label for each node
        G.add_node(node_id, **row.to_dict())

    # Create a unique mapping from (stepNum, ObjectNum) to node index
    node_mapping = {(row['ImageNumber'], row['ObjectNumber']): idx for idx, row in node_index_df.iterrows()}
    contact_edges = []
    
    # Add edges, considering contact edges
    for _, row in edge_features_df.iterrows():
        ##if row['Relationship'] == 'Neighbors':
        step_num_1 = row['First Image Number']
        step_num_2 = row['Second Image Number']
        node1 = row['First Object Number']
        node2 = row['Second Object Number']

        if (step_num_1, node1) in node_mapping and (step_num_2, node2) in node_mapping:
            if not G.has_edge(node_mapping[(step_num_1, node1)], node_mapping[(step_num_2, node2)]):
                node1_idx = node_mapping[(step_num_1, node1)]
                node2_idx = node_mapping[(step_num_2, node2)]
                G.add_edge(node1_idx, node2_idx, edge_type = 'contact')
                G.add_edge(node2_idx, node1_idx, edge_type = 'contact')
                contact_edges.append((node1_idx, node2_idx))
                contact_edges.append((node2_idx, node1_idx))

    # Create lineage mapping
    lineage_mapping = {(row['id'], row['parent_id'], row['ImageNumber']): idx for idx, row in node_index_df.iterrows()}
    lineage_edges = []
    
    # Add directed lineage edges based on the lineage mapping
    for key, node1_idx in lineage_mapping.items():
        id, parent_id, step_num = key
        if parent_id == 0:
            continue  # Skip if parent_id is zero
        if (id, id, step_num + 1) in lineage_mapping:
            node2_idx = lineage_mapping[(id, id, step_num + 1)]
            G.add_edge(node1_idx, node2_idx, edge_type='lineage')
            lineage_edges.append((node1_idx, node2_idx))
        elif (id, parent_id, step_num + 1) in lineage_mapping:
            node2_idx = lineage_mapping[(id, parent_id, step_num + 1)]
            G.add_edge(node1_idx, node2_idx, edge_type='lineage')
            lineage_edges.append((node1_idx, node2_idx))
        else:
            for parent_key, node2_idx in lineage_mapping.items():
                parent_id_key, _, step_num_key = parent_key
                if parent_id_key == parent_id and step_num_key == step_num - 1:
                    G.add_edge(node2_idx, node1_idx, edge_type='lineage')
                    lineage_edges.append((node2_idx, node1_idx))
                    break

    x = torch.tensor([flatten_node_attributes(G.nodes[node]) for node in G.nodes()], dtype=torch.float)
    contact_edge_index = torch.tensor(contact_edges, dtype=torch.long).t().contiguous()
    lineage_edge_index = torch.tensor(lineage_edges, dtype=torch.long).t().contiguous()

    # print(x.shape)  # all features in a form of vectors has been spread out as individual columns

    data =  Data(x, contact_edge_index=contact_edge_index, lineage_edge_index=lineage_edge_index)
    
    return data


if __name__ == '__main__':

    # 31 features
    include_columns = ['ImageNumber', #
                       'ObjectNumber', #
                       'id', #
                       'parent_id', #
                       'AreaShape_Area', #
                       'AreaShape_MajorAxisLength', #
                       'AreaShape_Orientation', #
                       'cellAge', #
                       'LifeHistory', #
                       'TrajectoryX', #
                       'TrajectoryY', #
                       'Direction_of_Motion', #
                       'Motion_Alignment_Angle', #
                       'Source_Neighbor_Avg_TrajectoryX', #
                       'Source_Neighbor_Avg_TrajectoryY', #
                       'Division_TimeStep', #
                       'Daughter_Mother_Length_Ratio', #
                       'Total_Daughter_Mother_Length_Ratio', #
                       'Max_Daughter_Mother_Length_Ratio', #
                       'Daughter_Avg_TrajectoryX', #
                       'Daughter_Avg_TrajectoryY', #
                       'Neighbor_Difference_Count', #
                       'Neighbor_Shared_Count', #
                       'Average_Length', #
                       'Velocity', #
                       'Instant_Velocity', #
                       'Average_Instant_Velocity', #
                       'Elongation_Rate', #
                       'Instant_Elongation_Rate', #
                       'startVol', #
                       'targetVol', #
                       'Prev_MajorAxisLength', #
                       'Bacterium_Movement', #
                       'dir_1', 'dir_2', #
                       ]
    
    final_path = "D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/2025-08-20/CVIS_S1_minmaxtest.pkl"  ### training mean/std path
    print(final_path)
    with open(final_path, "rb") as file:
        values = pk.load(file) 
    ##print(values)  
    gamma_mean = values["gamma mean"] 
    gamma_std = values["gamma std"] 
    reg_param_mean = values["reg_param mean"] 
    reg_param_std = values["reg_param std"]  
    mean_dict = values["feature mean"] 
    std_dict = values["feature std"] 
    gamma_min = values["gamma min"] 
    gamma_max = values["gamma max"] 
    reg_param_min = values["reg_param min"] 
    reg_param_max = values["reg_param max"]  

    print("log(gamma) mean: ", gamma_mean)
    print("log(gamma) std: ", gamma_std)
    print("log(reg_param) mean: ", reg_param_mean)
    print("log(reg_param) std: ", reg_param_std)
    ##print("feature mean: ", mean_dict)
    ##print("feature std: ", std_dict)
    print("log(gamma) min: ", gamma_min)
    print("log(gamma) max: ", gamma_max)
    print("log(reg_param) min: ", reg_param_min)
    print("log(reg_param) max: ", reg_param_max)
    

    feature_path = "D:\Projects\GNN Research\Data Files\_exp_tr_csv_data_sampling"
    neighbor_path = "D:\Projects\GNN Research\Data Files\_exp_csv_data_sampling"

    feature_files = [os.path.join(feature_path, f) for f in os.listdir(feature_path)]
    neighbor_files = [os.path.join(neighbor_path, f) for f in os.listdir(neighbor_path)]
    
    graph_id_list = [f.split("Trackrefiner")[0] for f in os.listdir(feature_path)]

    data_tuple = {}

    for g in graph_id_list:
        graph_id = g
        csv1_path = [f for f in feature_files if g in f][0]
        csv2_path = [n for n in neighbor_files if g in n][0]
        data_tuple[graph_id] = load_exp_graph_data(graph_id, csv1_path, csv2_path, include_columns)
        print(data_tuple[graph_id])
    ##breakpoint()

    # save the graph data
    folder_path = "D:/Projects/GNN Research/Data Files/_exp_graph_data_sampling/"
    os.makedirs(folder_path, exist_ok=True)
    g_data_path = os.path.join(folder_path, "CVIS_S1_realexp.pkl")
    with open(g_data_path, 'wb') as f:
        pk.dump(data_tuple, f)

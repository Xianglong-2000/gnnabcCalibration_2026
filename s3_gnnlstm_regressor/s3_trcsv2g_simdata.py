import os
import time
import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import random
from itertools import combinations, permutations, product
from torch.utils.data import DataLoader
from torch.optim import Adam
import pickle as pk
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

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


def load_graph_data(graph_ID, node_features_file, neighbours_edge_file, include_columns,
                    gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict):

    # Create the y vector, y = [gamma, reg_param]
    gamma = graph_ID.split("_")[1]
    alpha = graph_ID.split("_")[4]
    ##y1 = (math.log10(float(gamma)) - gamma_mean) / gamma_std  ### standard normalization
    ##y2 = (math.log10(float(alpha)) - reg_param_mean) / reg_param_std  ### standard normalization
    y1 = math.log10(float(gamma))
    y2 = math.log10(float(alpha))

    y = torch.tensor([y1, y2], dtype=torch.float)

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

    node_features_df = node_features_df[include_columns]
    node_features_df = node_features_df.fillna(node_features_df.mean(numeric_only=True))

    node_index_df = node_features_df.iloc[:, :4]
    node_features_df = node_features_df.iloc[:, 4:]

    df_normalized = node_features_df.copy()

    # Standard normalization
    binary_columns = ["divideFlag", "Unexpected_End", "Unexpected_Beginning"]
    for col in node_features_df.columns:
        if col not in binary_columns:
            df_normalized[col] = (node_features_df[col] - mean_dict[col]) / std_dict[col]
            ##print(mean_dict[col], std_dict[col])
    ##print(df_normalized["divideFlag"])
    ##print(df_normalized.head())

    node_features_df = df_normalized

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
        if row['Relationship'] == 'Neighbors':
            step_num_1 = row['First Image Number']
            step_num_2 = row['Second Image Number']
            node1 = row['First Object Number']
            node2 = row['Second Object Number']

            if (step_num_1, node1) in node_mapping and (step_num_2, node2) in node_mapping:
                if not G.has_edge(node_mapping[(step_num_1, node1)], node_mapping[(step_num_2, node2)]):
                    node1_idx = node_mapping[(step_num_1, node1)]
                    node2_idx = node_mapping[(step_num_2, node2)]
                    G.add_edge(node1_idx, node2_idx, edge_type='contact')
                    G.add_edge(node2_idx, node1_idx, edge_type='contact')
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
    data = Data(x, contact_edge_index=contact_edge_index, lineage_edge_index=lineage_edge_index, y=y)

    return data

def get_mean_and_std(param_dict, include_columns):

    all_x = np.empty((0, 70))
    gamma_list = []
    reg_param_list = []
    for param in param_dict.keys():

        print("stacking ", param)

        gamma = param.split("_")[1]
        alpha = param.split("_")[4]
        y1 = math.log10(float(gamma))
        y2 = math.log10(float(alpha))
        gamma_list.append(y1)
        reg_param_list.append(y2)

        df = pd.read_csv(param_dict[param][0])

        for col in df.columns:
            df[col] = df[col].map(smart_cast)
        df['dir'] = df['dir'].apply(eval).apply(lambda x: tuple(map(float, x)))
        df = df.fillna(df.mean(numeric_only=True))
        vec_df = pd.DataFrame(df['dir'].tolist(), columns=['dir_1', 'dir_2'])
        df = pd.concat([df.drop(columns=['dir']), vec_df], axis=1)

        x = df.to_numpy()
        all_x_last = all_x.copy()
        all_x = np.concatenate((all_x_last, x), axis=0)

    columns = list(pd.read_csv(param_dict[list(param_dict.keys())[0]][0]).drop(columns=['dir']).columns)+['dir_1', 'dir_2']
    df = pd.DataFrame(all_x, columns=columns)
    df = df[include_columns]
    print("stacked data size: ", df.shape)

    # standard normalization for both inputs and outputs
    gamma_mean = np.mean(gamma_list)
    gamma_std = np.std(gamma_list)
    reg_param_mean = np.mean(reg_param_list)
    reg_param_std = np.std(reg_param_list)
    mean_dict = df.mean().to_dict()
    std_dict = df.std().to_dict()
    print("log(gamma) mean: ", gamma_mean)
    print("log(gamma) std: ", gamma_std)
    print("log(reg_param) mean: ", reg_param_mean)
    print("log(reg_param) std: ", reg_param_std)
    print("feature mean: ", mean_dict)
    print("feature std: ", std_dict)
    return gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict

def test1():

    sim_date = "2025-09-25"
    parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/{sim_date}/"
    iterations = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if
                  os.path.isdir(os.path.join(parent_dir, f))]
    param_dict = {}
    for i in iterations:
    ##for i in [iterations[0]]:  ### for a test only
        sub_folder_name = [f for f in os.listdir(i) if os.path.isdir(os.path.join(i, f))][0]
        new_name = sub_folder_name.replace("=", "_").replace("_iter=", "-Average-")
        csv_path_1 = f"{i}/{sub_folder_name}/Trackrefiner.Objects properties_Average_analysis.csv"
        j = i.replace("_sim_tr_csv_data_gnn", "_sim_csv_data_gnn")
        csv_path_2 = f"{j}/{sub_folder_name}/Object relationships.csv"
        param_dict[new_name] = [csv_path_1, csv_path_2]

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

    num_features = len(include_columns)-4
    print("num of features: ", num_features )
    gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict = get_mean_and_std(param_dict, include_columns)

    # save mean and std values
    values = {"gamma mean": gamma_mean,
              "gamma std": gamma_std,
              "reg_param mean": reg_param_mean,
              "reg_param std": reg_param_std,
              "feature mean": mean_dict,
              "feature std": std_dict}

    folder_path = f"D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/{sim_date}/"
    os.makedirs(folder_path, exist_ok=True)
    final_path = os.path.join(folder_path, "CVIS_S3.pkl")
    with open(final_path, 'wb') as f:
        pk.dump(values, f)

    # generate graphs
    start_time = time.time()
    data_tuple = {}

    for key in param_dict.keys():
        data_tuple[key] = load_graph_data(key, param_dict[key][0], param_dict[key][1], include_columns,
                                          gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict)
        print(data_tuple[key])

    end_time = time.time()
    print(f"Computing time: {(end_time - start_time) / 3600:.2f} hrs")
    print()

    # save graphs
    folder_path = f"D:/Projects/GNN Research/Data Files/_sim_graph_data_gnn/{sim_date}/"
    os.makedirs(folder_path, exist_ok=True)
    g_data_path = os.path.join(folder_path, "CVIS_S3.pkl")
    with open(g_data_path, 'wb') as f:
        pk.dump(data_tuple, f)

if __name__ == "__main__":
    test1()




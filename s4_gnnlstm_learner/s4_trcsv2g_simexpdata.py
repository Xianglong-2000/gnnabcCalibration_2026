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
import matplotlib.pyplot as plt
import re
from collections import defaultdict
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

def split_node_feature_csv(csv_path):
    """
    csv_path: "D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/2025-08-26/iteration 000/gamma=37.6_reg_param=1.24_iter=0/Trackrefiner.Objects properties_Average_analysis.csv"
    return: the path where sub-csv files are saved
    """
    df = pd.read_csv(csv_path)
    split_column = "ImageNumber"
    file_path = csv_path.replace("_sim_tr_csv_data_gnn", "_sim_tr_csv_splitted_data_gnn").replace(
        "/Trackrefiner.Objects properties_Average_analysis.csv", "/")
    for value, group in df.groupby(split_column):
        sub_csv_path = file_path + f"node_feature_data/ImageNumber={value}.csv"
        os.makedirs(file_path + "node_feature_data", exist_ok=True)
        group.to_csv(sub_csv_path, index=False)
    return file_path + "node_feature_data"


def split_node_neighbor_csv(csv_path):
    """
    csv_path: "D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/2025-08-26/iteration 000/gamma=37.6_reg_param=1.24_iter=0/Object relationships.csv"
    return: the path where sub-csv files are saved
    """
    df = pd.read_csv(csv_path)
    split_column = "First Image Number"
    file_path = csv_path.replace("_sim_csv_data_gnn", "_sim_tr_csv_splitted_data_gnn").replace(
        "/Object relationships.csv", "/")
    for value, group in df.groupby(split_column):
        sub_csv_path = file_path + f"node_neighbor_data/ImageNumber={value}.csv"
        os.makedirs(file_path + "node_neighbor_data", exist_ok=True)
        group.to_csv(sub_csv_path, index=False)
    return file_path + "node_neighbor_data"

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
    ##y2 = (math.log10(float(alpha)) - reg_param_mean) / reg_param_std
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
    ##node_features_df = node_features_df.fillna(node_features_df.mean())
    for col in node_features_df.columns:  ### some sub-graphs have columns with all nan values so we need conditional treatments
        mean_val = node_features_df[col].mean()
        if pd.isna(mean_val):
            node_features_df[col] = node_features_df[col].fillna(0)
            print("empty col: ", col, node_features_df[col].dtype)
        else:
            node_features_df[col] = node_features_df[col].fillna(mean_val)

    node_index_df = node_features_df.iloc[:, :2]
    node_features_df = node_features_df.iloc[:, 2:]

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

    x = torch.tensor([flatten_node_attributes(G.nodes[node]) for node in G.nodes()], dtype=torch.float)
    edge_index = torch.tensor(contact_edges, dtype=torch.long).t().contiguous()
    data = Data(x, edge_index=edge_index, y=y)

    return data


def test1():

    sim_date = "2025-08-24"
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

    num_features = len(include_columns)-2
    print("num of features: ", num_features )

    # load training mean/std of features and targets
    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/2025-09-25/" 
    final_path = parent_dir + "CVIS_S4.pkl" 
    print(final_path) 
    with open(final_path, "rb") as file:  
        values = pk.load(file)  
    print(values)  
    gamma_mean = values["gamma mean"]  
    gamma_std = values["gamma std"]  
    reg_param_mean = values["reg_param mean"]  
    reg_param_std = values["reg_param std"] 
    mean_dict = values["feature mean"]  
    std_dict = values["feature std"]  

    # generate graphs
    start_time = time.time()

    for key in param_dict.keys():
        print("Graph ID: ", key)
        data_tuple = {}
        feature_path = split_node_feature_csv(param_dict[key][0])
        neighbor_path = split_node_neighbor_csv(param_dict[key][1])
        feature_files = [f for f in os.listdir(feature_path)]
        neighbor_files = [f for f in os.listdir(neighbor_path)]
        print(f"number of feature files = {len(feature_files)}; number of neighbor files = {len(neighbor_files)}")
        folder_path = feature_path.replace("_sim_tr_csv_splitted_data_gnn", "_sim_graph_splitted_data_gnn")
        os.makedirs(folder_path, exist_ok=True)
        for f in feature_files:
            if f in neighbor_files:  ### filter out the first 25 csv files because there's only one single cell without any edges
                csv_path_1 = os.path.join(feature_path, f)
                csv_path_2 = os.path.join(neighbor_path, f)
                data_tuple[f"{key}_{f}"] = load_graph_data(key, csv_path_1, csv_path_2, include_columns,
                                          gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict)
                print(f"{key}_{f}: ", data_tuple[f"{key}_{f}"])

        g_data_path = os.path.join(folder_path, "CVIS_S4_simexp.pkl")
        with open(g_data_path, 'wb') as f:
            pk.dump(data_tuple, f)

    end_time = time.time()
    print(f"Computing time: {(end_time - start_time)/3600:.2f} hrs")
    print()

if __name__ == "__main__":
    test1()




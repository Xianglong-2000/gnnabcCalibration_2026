import os
import sys
import time
import pandas as pd
import numpy as np
import random
import math
import re
import matplotlib.pyplot as plt
import pickle as pk
from collections import defaultdict
from datetime import datetime
from itertools import combinations, permutations, product
from tqdm import tqdm

import networkx as nx
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset

import models_gatlstmregressor


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

def get_param_dict(parent_dir_tr: str, parent_dir: str):

    # parent_dir_tr = "D:\Projects\GNN Research\Data Files\_sim_tr_csv_data_gnn\2025-07-02"  ## node features
    # parent_dir = "D:\Projects\GNN Research\Data Files\_sim_csv_data_gnn\2025-07-02"  ## neighbours edge

    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    ##folders = ["iteration 001"]  ## for a test
    ## print(len(folders))
    pattern = re.compile(r"gamma=(\d+(?:\.\d+)?)_reg_param=(\d+(?:\.\d+)?)_iter=(\d+)")
    param_dict = {}
    for f in folders:
        sub_parent_dir_tr = parent_dir_tr + "/" + f
        sub_parent_dir = parent_dir + "/" + f
        sub_folders = [f for f in os.listdir(sub_parent_dir) if os.path.isdir(os.path.join(sub_parent_dir, f))]
        ##sub_folders = ["gamma=327.89_reg_param=23.41_iter=1"]  ## for a test
        for ff in sub_folders:
            match = pattern.search(ff)
            gamma, reg_param, average = match.groups()
            counter = float(average)
            if counter <= 9:
                param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-00{average}"
            elif (counter > 9)&(counter <= 99):
                param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-0{average}"
            elif (counter > 99)&(counter <= 999):
                param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-{average}"
            csv1_path = sub_parent_dir_tr + "/" + ff + "/Trackrefiner.Objects properties_Average_analysis.csv"  # node features
            csv2_path = sub_parent_dir + "/" + ff + "/Object relationships.csv"  # neighbours
            param_dict[param_pair_index] = [csv1_path, csv2_path]

    return param_dict  # {"gamma_reg_param_Average": [csv1 path, csv2 path]}


def flatten_node_attributes(node_attr):
        """Flatten the node attributes to ensure all values are numeric."""
        flat_attrs = []
        for attr in node_attr.values():
            if isinstance(attr, (tuple, list)):
                flat_attrs.extend(attr)
            else:
                flat_attrs.append(attr)
        return flat_attrs


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
    split_list = graph_ID[:-12].split('_')
    y1 = (math.log10(float(split_list[1])) - gamma_mean) / gamma_std   ### standard normalization
    y2 = (math.log10(float(split_list[4])) - reg_param_mean) / reg_param_std
    y = torch.tensor([y1,y2], dtype=torch.float)
    
    # Read though the csv files and create the df
    node_features_df = pd.read_csv(node_features_file, dtype=str)
    neighbours_edge_df = pd.read_csv(neighbours_edge_file, dtype=str)
    edge_features_df = neighbours_edge_df

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

    node_index_df = node_features_df.iloc[:, :2]  ### replace 2 with 4 if we have 2 edge types
    node_features_df = node_features_df.iloc[:, 2:]  ### replace 2 with 4 if we have 2 edge types

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
    print("num of nan values: ", node_features_df.isna().sum().sum())  ### show num of nan values across the whole dataframe
    print("num of cols with nan:", node_features_df.isna().any().sum())  ### show num of cols with nan across all cols
    print("cols with nan:", node_features_df.columns[node_features_df.isna().any()].tolist())  ### show which cols with nan across all cols

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
                    G.add_edge(node1_idx, node2_idx, edge_type = 'contact')
                    G.add_edge(node2_idx, node1_idx, edge_type = 'contact')
                    contact_edges.append((node1_idx, node2_idx))
                    contact_edges.append((node2_idx, node1_idx))

    x = torch.tensor([flatten_node_attributes(G.nodes[node]) for node in G.nodes()], dtype=torch.float)
    edge_index = torch.tensor(contact_edges, dtype=torch.long).t().contiguous()

    data =  Data(x, edge_index=edge_index, y=y)
    return data


def get_mean_and_std(param_dict, include_columns):

    all_x = np.empty((0, 70))
    gamma_list = []
    reg_param_list = []
    for param in param_dict.keys():

        print("stacking ", param)

        split_list = param[:-12].split('_')
        #print(split_list)
        y1 = math.log10(float(split_list[1]))
        y2 = math.log10(float(split_list[4]))
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
    gamma_min = np.min(gamma_list)
    gamma_max = np.max(gamma_list)
    reg_param_min = np.min(reg_param_list)
    reg_param_max = np.max(reg_param_list)

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

    return gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict, gamma_min, gamma_max, reg_param_min, reg_param_max


if __name__ == "__main__":

    dt_sim = "2025-08-20"

    parent_dir_tr = f"D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/{dt_sim}"
    parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/{dt_sim}"
    param_dict = get_param_dict(parent_dir_tr, parent_dir)
    print(len(param_dict.keys()))

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
    print("number of node features: ", num_features)

    gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict, gamma_min, gamma_max, reg_param_min, reg_param_max = get_mean_and_std(param_dict, include_columns)

    values = {"gamma mean": gamma_mean,
              "gamma std": gamma_std,
              "reg_param mean": reg_param_mean,
              "reg_param std": reg_param_std,
              "feature mean": mean_dict,
              "feature std": std_dict,
              "gamma min": gamma_min,
              "gamma max": gamma_max,
              "reg_param min": reg_param_min,
              "reg_param max": reg_param_max,}
    folder_path = "D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/2025-08-20/"
    os.makedirs(folder_path, exist_ok=True)
    final_path = os.path.join(folder_path, "CVIS_S2_minmaxtest.pkl")
    with open(final_path, 'wb') as f:
        pk.dump(values, f)

    breakpoint()

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

        g_data_path = os.path.join(folder_path, "CVIS_S2.pkl")
        with open(g_data_path, 'wb') as f:
            pk.dump(data_tuple, f)

    end_time = time.time()
    print(f"Computing time: {(end_time - start_time)/3600:.2f} hrs")
    print()

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

def get_param_dict(parent_dir: str):

    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    pattern = re.compile(r"gamma=(\d+(?:\.\d+)?)_reg_param=(\d+(?:\.\d+)?)_iter=(\d+)")
    param_dict = {}
    for f in folders:
        match = pattern.search(f)
        gamma, reg_param, average = match.groups()
        counter = float(average)
        if counter <= 9:
            param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-00{average}"
        elif (counter > 9)&(counter <= 99):
            param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-0{average}"
        elif (counter > 99)&(counter <= 999):
            param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-{average}"
        csv1_path = parent_dir + "/" + f + "/Trackrefiner/Trackrefiner.Objects_properties_Average_analysis.csv"
        csv2_path = parent_dir + "/" + f + "/Object_relationships.csv"
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
    split_list = graph_ID[:-12].split('_')
    # print(split_list)
    y1 = (math.log(float(split_list[1])) - gamma_mean) / gamma_std   ### standard normalization
    y2 = (math.log(float(split_list[4])) - reg_param_mean) / reg_param_std

    y = torch.tensor([y1,y2], dtype=torch.float)
    
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
    #print(node_features_df.head())

    node_index_df = node_features_df.iloc[:, :4]
    node_features_df = node_features_df.iloc[:, 4:]

    df_normalized = node_features_df.copy()

    # Standard normalization
    binary_columns = ["divideFlag", "Unexpected_End", "Unexpected_Beginning"]
    for col in node_features_df.columns:
        if col not in binary_columns:
            df_normalized[col] = (node_features_df[col] - mean_dict[col])/std_dict[col]

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

    data =  Data(x, contact_edge_index=contact_edge_index, lineage_edge_index=lineage_edge_index, y=y)
    return data


def get_mean_and_std(param_dict, include_columns):

    all_x = np.empty((0, 74))
    gamma_list = []
    reg_param_list = []
    empty_files = []
    for param in param_dict.keys():

        print("stacking ", param)

        split_list = param[:-12].split('_')
        #print(split_list)
        y1 = math.log(float(split_list[1]))
        y2 = math.log(float(split_list[4]))
        gamma_list.append(y1)
        reg_param_list.append(y2)

        try:  # some files empty bc skipped over in CM/OP/TR parts
            df = pd.read_csv(param_dict[param][0])
        except Exception as e:
            empty_files.append(param)
            continue

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

    print("ln(gamma mean): ", gamma_mean)
    print("ln(gamma std): ", gamma_std)
    print("ln(alpha mean): ", reg_param_mean)
    print("ln(alpha std): ", reg_param_std)
    ##print("feature mean: ", mean_dict)
    ##print("feature std: ", std_dict)
    print("ln(gamma min): ", gamma_min)
    print("ln(gamma max): ", gamma_max)
    print("ln(alpha min): ", reg_param_min)
    print("ln(alpha max): ", reg_param_max)

    return gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict, gamma_min, gamma_max, reg_param_min, reg_param_max, empty_files


if __name__ == "__main__":

    parent_dir = "/work/x5bai/project/Data_Files/_sim_tr_csv_data_gnn/2025-11-11"
    param_dict = get_param_dict(parent_dir)  
    print(len(param_dict.keys()))

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
    print("number of node features: ", num_features)

    gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict, gamma_min, gamma_max, reg_param_min, reg_param_max, empty_files = get_mean_and_std(param_dict, include_columns)

    # do NOT run this part everytime
    values = {"ln(gamma mean)": gamma_mean,
              "ln(gamma std)": gamma_std,
              "ln(reg_param mean)": reg_param_mean,
              "ln(reg_param std)": reg_param_std,
              "feature mean": mean_dict,
              "feature std": std_dict,
              "ln(gamma min)": gamma_min,
              "ln(gamma max)": gamma_max,
              "ln(alpha min)": reg_param_min,
              "ln(alpha max)": reg_param_max,}
    folder_path = "/work/x5bai/project/Data_Files/_sim_mean_std_values_gnn/2025-11-11/"
    os.makedirs(folder_path, exist_ok=True)
    final_path = os.path.join(folder_path, "S1_test_1111.pkl")
    with open(final_path, 'wb') as f:
        pk.dump(values, f)

    ##breakpoint()

    # generate graphs
    start_time = time.time()
    data_tuple = {}

    for key in param_dict.keys():
        if key not in empty_files:  # some files empty bc skipped over in CM/OP/TR parts
            data_tuple[key] = load_graph_data(key, param_dict[key][0], param_dict[key][1], include_columns,
                                            gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict)
        else:
            continue
        print(data_tuple[key])

    # check the training duration
    end_time = time.time()
    print(f"Computing time: {(end_time - start_time)/3600:.2f} hrs")
    print()

    # save the graph data
    folder_path = "/work/x5bai/project/Data_Files/_sim_graph_data_gnn/2025-11-11/"
    os.makedirs(folder_path, exist_ok=True)
    g_data_path = os.path.join(folder_path, "S1_test_1111.pkl")
    with open(g_data_path, 'wb') as f:
        pk.dump(data_tuple, f)


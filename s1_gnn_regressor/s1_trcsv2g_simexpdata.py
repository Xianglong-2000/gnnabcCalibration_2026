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


def get_param_dict(parent_dir_tr: str, parent_dir: str):

    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    ##folders = ["iteration 000"]  ## for a test
    pattern = re.compile(r"gamma=(\d+(?:\.\d+)?)_reg_param=(\d+(?:\.\d+)?)_iter=(\d+)")
    param_dict = {}
    for f in folders:
        sub_parent_dir_tr = parent_dir_tr + "/" + f
        sub_parent_dir = parent_dir + "/" + f
        sub_folders = [f for f in os.listdir(sub_parent_dir) if os.path.isdir(os.path.join(sub_parent_dir, f))]
        ##sub_folders = ["gamma=182.89_reg_param=2.43_iter=0"]  ## for a test
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
    y1 = (math.log10(float(split_list[1])) - gamma_mean) / gamma_std   ### standard normalization
    y2 = (math.log10(float(split_list[4])) - reg_param_mean) / reg_param_std

    y = torch.tensor([y1,y2], dtype=torch.float)

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


if __name__ == "__main__":

    dt_sim = "2025-08-24"

    feature_csv_dir = f"D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/{dt_sim}"
    neighbor_csv_dir = f"D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/{dt_sim}"
    param_dict = get_param_dict(feature_csv_dir, neighbor_csv_dir)
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

    num_features = len(include_columns)
    print("num of node features: ", num_features)

    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/2025-08-20/"
    final_path = parent_dir + "CVIS_S1.pkl"
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

    start_time = time.time()
    data_tuple = {}

    for key in param_dict.keys():

        data_tuple[key] = load_graph_data(key, param_dict[key][0], param_dict[key][1], include_columns,
                                          gamma_mean, gamma_std, reg_param_mean, reg_param_std, mean_dict, std_dict)
        print(data_tuple[key])

    # check the training duration
    end_time = time.time()
    print(f"Computing time: {(end_time - start_time)/3600:.2f} hrs")
    print()

    # save the graph data
    folder_path = f"D:/Projects/GNN Research/Data Files/_sim_graph_data_gnn/{dt_sim}/"
    os.makedirs(folder_path, exist_ok=True)
    g_data_path = os.path.join(folder_path, "CVIS_S1_simexp.pkl")
    with open(g_data_path, 'wb') as f:
        pk.dump(data_tuple, f)

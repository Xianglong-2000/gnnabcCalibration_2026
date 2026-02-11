import os
import time
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
# from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import ml_dl_models


def get_param_dict(parent_dir_tr, parent_dir, features_targets_dir):

    # parent_dir_tr = "D:\Projects\GNN Research\Data Files\_sim_tr_csv_data_gnn\2025-07-02"  ## node features
    # parent_dir = "D:\Projects\GNN Research\Data Files\_sim_csv_data_gnn\2025-07-02"  ## neighbours edge

    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    pattern = re.compile(r"gamma=(\d+(?:\.\d+)?)_reg_param=(\d+(?:\.\d+)?)_iter=(\d+)")
    param_dict = {}
    for f in folders:
        sub_parent_dir_tr = parent_dir_tr + "/" + f
        sub_parent_dir = parent_dir + "/" + f
        sub_features_targets_dir = features_targets_dir + "/" + f
        sub_folders = [f for f in os.listdir(sub_parent_dir) if os.path.isdir(os.path.join(sub_parent_dir, f))]
        for ff in sub_folders:
            match = pattern.search(ff)
            gamma, reg_param, average = match.groups()
            counter = float(average)
            if counter <= 9:
                param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-0{average}"
            elif (counter > 9)&(counter <= 99):
                param_pair_index = f"gamma_{gamma}_reg_param_{reg_param}-Average-{average}"
            csv1_path = sub_parent_dir_tr + "/" + ff + "/Trackrefiner.Objects properties_Average_analysis.csv"  # node indices
            csv2_path = sub_parent_dir + "/" + ff + "/Object relationships.csv"  # neighbours
            csv3_path = sub_features_targets_dir + "/" + ff + "/Trackrefiner.Objects properties_Average_analysis.csv"  # features & targets
            param_dict[param_pair_index] = [csv1_path, csv2_path, csv3_path]
    return param_dict  # {"gamma_reg_param_Average": [csv1 path, csv2 path, csv3 path]}



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


def load_graph_data(node_index_file, neighbours_edge_file, preprocessed_data_file):

    # step 1: read though the csv files and create the dfs
    node_index_df = pd.read_csv(node_index_file, dtype=str)
    neighbours_edge_df = pd.read_csv(neighbours_edge_file, dtype=str)
    preprocessed_data_df = pd.read_csv(preprocessed_data_file, dtype=str)

    # step 2: turn everything from string type to float
    for col in node_index_df.columns:
        node_index_df[col] = node_index_df[col].map(smart_cast)
    for col in neighbours_edge_df.columns:
        neighbours_edge_df[col] = neighbours_edge_df[col].map(smart_cast)
    for col in preprocessed_data_df.columns:
        preprocessed_data_df[col] = preprocessed_data_df[col].map(smart_cast)

    # step 3: split data to features and targets
    node_features_df = preprocessed_data_df.iloc[:, :-2]
    targets_df = preprocessed_data_df.iloc[:, -2:]

    # step 4: take targets into a tensor
    y1 = targets_df["gamma"][0]
    y2 = targets_df["reg_param"][0]
    y = torch.tensor([[y1],
                      [y2]], dtype=torch.float)

    # step 5: get lineage edge df and concat it to contact edge df
    parents_edge_df = generate_parent_df(node_index_df)
    edge_features_df = pd.concat([parents_edge_df, neighbours_edge_df])


    # print(node_features_df.dtypes)
    # print(node_features_df.head(3))
    # print(len(preprocessed_data_df.columns))
    # print(len(node_features_df.columns))

    # step 6: create the graph
    G = nx.MultiDiGraph()

    # step 7: add nodes with their features as attributes and label them by their index
    for idx, row in node_features_df.iterrows():
        node_id = idx  # Use the index as a unique label for each node
        G.add_node(node_id, **row.to_dict())

    # step 8: create a unique mapping from (stepNum, ObjectNum) to node index
    node_mapping = {(row['ImageNumber'], row['ObjectNumber']): idx for idx, row in node_index_df.iterrows()}
    contact_edges = []
    
    # step 9: add edges, considering contact edges
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

    # step 10: create lineage mapping
    lineage_mapping = {(row['id'], row['parent_id'], row['ImageNumber']): idx for idx, row in node_index_df.iterrows()}
    lineage_edges = []
    
    # step 11: add directed lineage edges based on the lineage mapping
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

    # print([G.nodes[node].keys() for node in G.nodes()])
    # print([len(G.nodes[node].keys()) for node in G.nodes()])
    # print([G.nodes[node].values() for node in G.nodes()])

    # step 12: put everything together
    x = torch.tensor([flatten_node_attributes(G.nodes[node]) for node in G.nodes()], dtype=torch.float)
    contact_edge_index = torch.tensor(contact_edges, dtype=torch.long).t().contiguous()
    lineage_edge_index = torch.tensor(lineage_edges, dtype=torch.long).t().contiguous()
    data =  Data(x, contact_edge_index=contact_edge_index, lineage_edge_index=lineage_edge_index, y=y)
    
    return data




def split_data(dataset:list) -> tuple:
    
    random.shuffle(dataset)

    train_size = int(len(dataset)*train_split)
    test_size = int(len(dataset)*test_split)
    val_size = int(len(dataset)*val_split)

    final_train_data = dataset[:train_size]
    final_test_data = dataset[train_size:train_size+val_size]
    final_val_data = dataset[train_size+val_size:]
    
    return final_train_data, final_test_data, final_val_data



def ssr_loss(pred, target):
    return torch.sum((pred - target) ** 2)



def tss_baseline(target):
    return torch.sum((target - target.mean())**2)



def R_2(ssr, tss):
    return 1 - ssr/tss



def mse_loss(pred, target):
    return ((target - pred)**2).mean()



# Train the model using SSR and deriving R^2
def model_train(in_channels_model, hidden_channels_model, feature_embedding_size, model_learning_rate, num_epoch, 
                train_loader, val_loader, patience = 20):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ml_dl_models.GAT(in_channels = in_channels_model, hidden_channels = hidden_channels_model, out_channels=feature_embedding_size).to(device)
    optimizer = Adam(model.parameters(), lr=model_learning_rate)
    print(model)

    start_time = time.time()
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    best_model_state = None

    epoch_num_list = []
    train_loss_list = []
    val_loss_list = []

    train_R_2_list = []
    val_R_2_list = []

    for epoch in range(1, num_epoch+1):
    
        # Training
        model.train()
        total_train_loss = 0
        total_train_tss = 0
        for batch in train_loader:
            batch = batch.to(device)

            # print("NaNs in input x:", torch.isnan(batch.x).sum().item())
            # print("Infs in input x:", torch.isinf(batch.x).sum().item())
            # print("NaNs in target y:", torch.isnan(batch.y).sum().item())
            # print("Infs in target y:", torch.isinf(batch.y).sum().item())
            # print(batch)

            optimizer.zero_grad()
            pred = model(batch)

            # print(pred)

            loss = ssr_loss(pred, batch.y.view(-1))
            target = batch.y.view(-1)
            TSS = torch.sum((target - target.mean())**2)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_tss += TSS

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_tss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = ssr_loss(pred, batch.y.view(-1))
                target = batch.y.view(-1)
                TSS = torch.sum((target - target.mean())**2)
                total_val_loss += loss.item()
                total_val_tss += TSS

        epoch_num_list.append(epoch)
        train_loss_list.append(total_train_loss)
        val_loss_list.append(total_val_loss)
    
        train_R_2 = R_2(total_train_loss, total_train_tss)
        val_R_2 = R_2(total_val_loss, total_val_tss)
    
        train_R_2_list.append(train_R_2)
        val_R_2_list.append(val_R_2)
    
        print(f"Epoch {epoch:03d}, Train SSR: {total_train_loss:.4f}, Train R_2: {train_R_2:.4f}, Val SSR: {total_val_loss:.4f}, Val R_2: {val_R_2:.4f}")

        # Early stopping logic
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_epoch = epoch
            counter = 0
            best_model_state = model.state_dict()  # Save the best model
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break

    # Load the best model before testing
    model.load_state_dict(best_model_state)

    # check the training duration
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    # Visualize the training/validation loss vs the epoch number with R^2
    x = epoch_num_list
    y1 = train_R_2_list
    y2 = val_R_2_list
    plt.plot(x, y1, label='Training loss', linestyle='-')
    plt.plot(x, y2, label='Validation loss', linestyle='--')
    plt.xlabel('Epoch Number')
    plt.ylabel('Relative Loss Change')
    plt.title('Training/Validation R^2 Score vs Epoch Number')
    plt.legend()
    plt.show()
    
    return model



def model_test(model, test_loader) -> None:
    # only one epoch needed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    total_test_loss = 0
    total_test_tss = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = ssr_loss(pred, batch.y.view(-1))
            target = batch.y.view(-1)
            TSS = torch.sum((target - target.mean())**2)
            total_test_loss += loss.item()
            total_test_tss += TSS
        
    total_test_R_2 = R_2(total_test_loss, total_test_tss)
    print(f"Testing SSR: {total_test_loss:.4f}, Testing R_2: {total_test_R_2:.4f}")



# Here we want to save our trained model
def model_save(model, path) -> None:

    torch.save(model.state_dict(), path)
    
    print("Model saved.")



if __name__ == "__main__":

    parent_dir_tr = "D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/2025-07-25"  ## for node index
    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/2025-07-25"  ## for contact edge
    features_targets_dir = "D:/Projects/GNN Research/Data Files/_sim_tr_csv_feature_selected_data_gnn/2025-07-25"  ## for features and targets
    param_dict = get_param_dict(parent_dir_tr, parent_dir, features_targets_dir)
    # print(param_dict)
    print(len(param_dict.keys()))

    start_time = time.time()
    data_tuple = {}
    counter = 0

    for key in param_dict.keys():
        data_tuple[key] = load_graph_data(param_dict[key][0], param_dict[key][1], param_dict[key][2])
        # print(param_dict[key][0])
        # print(param_dict[key][1])
        # print(data_tuple[key])
        # print(data_tuple[key].y)
        counter += 1
        if counter == 1:
            print('Processing...')
            print(f"{key}: {data_tuple[key]}")
        elif counter == len(list(param_dict.keys())):
            print(f"{key}: {data_tuple[key]}")
            print('...Done')
        else:
            print(f"{key}: {data_tuple[key]}")

    # print(data_tuple)

    # check the training duration
    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds")
    print()

    # Split data
    train_split, test_split, val_split = 0.7, 0.1, 0.2
    dataset = list(data_tuple.copy().values())
    final_train_data, final_test_data, final_val_data = split_data(dataset)
    # print(final_train_data)

    # Set up batches
    batch_train_size, batch_test_size, batch_val_size = 25, 10, 10
    train_loader = DataLoader(final_train_data, batch_size=batch_train_size, shuffle=True)
    test_loader = DataLoader(final_test_data, batch_size=batch_test_size, shuffle=True)
    val_loader = DataLoader(final_val_data, batch_size=batch_val_size, shuffle=True)


    # Train
    in_channels_model = 20
    hidden_channels_model = 20
    feature_embedding_size = 2
    model_learning_rate = 0.001
    num_epoch = 300
    model = model_train(in_channels_model, hidden_channels_model, feature_embedding_size,
                model_learning_rate, num_epoch, train_loader, val_loader)

    # Test
    model_test(model, test_loader)

    # Save trained model parameters
    # folder_name = "D:/Projects/GNN Research/Data Files/_model_data_new/"
    # dt_pickles = "2025-07-02"
    # save_path = folder_name + "trained_model_" + dt_pickles + ".pth"
    # model_save(model, save_path)

import os
import sys
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
from torch.optim import Adam, AdamW
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

sys.path.append('/work/x5bai/project/Code_Files/s1/')
import s1_models


def split_data(dataset:list) -> tuple:
    
    random.shuffle(dataset)

    train_size = int(len(dataset)*train_split)
    test_size = int(len(dataset)*test_split)
    val_size = int(len(dataset)*val_split)

    final_train_data = dataset[:train_size]
    final_test_data = dataset[train_size:train_size+test_size]
    final_val_data = dataset[train_size+test_size:]
    
    return final_train_data, final_test_data, final_val_data

# Train the model using SSR and deriving R^2
def model_train(in_channels_model, hidden_channels_model, feature_embedding_size, model_learning_rate, num_epoch, 
                train_loader, val_loader, patience = 20):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = s1_models.GAT_Regressor(in_channels=in_channels_model, hidden_channels=hidden_channels_model,
                                       out_channels=feature_embedding_size).to(device)

    optimizer = AdamW(model.parameters(), lr=model_learning_rate, weight_decay=1e-4)

    print(model)

    start_time = time.time()
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    best_model_state = None

    epoch_num_list = []
    train_loss_list = []
    val_loss_list = []
    training_targets = []
    validating_targets = []

    for epoch in range(1, num_epoch+1):
        total_train_loss = 0
        total_val_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch).view(-1)
            target = batch.y
            if epoch == 1:
                training_targets.append(target)

            ##print("targets: ",target)
            ##print("predictions: ", pred)
            ##print()

            loss = F.mse_loss(pred, target, reduction="sum")
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)
                target = batch.y
                if epoch == 1:
                    validating_targets.append(target)

                loss = F.mse_loss(pred, target, reduction="sum")

                total_val_loss += loss.item()

        epoch_num_list.append(epoch)
        train_loss_list.append(total_train_loss/(2*len(final_train_data)))
        val_loss_list.append(total_val_loss/(2*len(final_val_data)))
    
        print(f"Epoch {epoch:03d}, Train MSE: {total_train_loss/(2*len(final_train_data)):.4f}, Val MSE: {total_val_loss/(2*len(final_val_data)):.4f}")

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
    print(f"Training time: {(end_time - start_time)/3600:.2f} hrs")

    # print the training and validating targets for distribution verification
    training_gamma = [v[0] for v in training_targets]
    training_alpha = [v[1] for v in training_targets]
    validating_gamma = [v[0] for v in validating_targets]
    validating_alpha = [v[1] for v in validating_targets]
    print("training_gamma = ", training_gamma)
    print("training_alpha = ", training_alpha)
    print("validating_gamma = ", validating_gamma)
    print("validating_alpha = ", validating_alpha)

    # comment it off for gnn feature selection only
    x = epoch_num_list
    y1 = train_loss_list
    y2 = val_loss_list
    plt.plot(x, y1, label='Training loss', linestyle='-')
    plt.plot(x, y2, label='Validation loss', linestyle='--')
    plt.xlabel('Epoch Number')
    plt.ylabel('MSE')
    plt.title('MSE vs Epoch Number')
    plt.legend()
    plt.show()
    
    return model


def model_test(model, test_loader) -> None:
    # only one epoch needed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_list = []
    p_list = []
    testing_targets = []
    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).view(-1)
            target = batch.y
            testing_targets.append(target)
            t_list.append(target)
            p_list.append(pred)
            ##print("targets: ",target)
            ##print("predictions: ", pred)

            loss = F.mse_loss(pred, target, reduction="sum")

            total_test_loss += loss.item()

    print(f"Testing MSE: {total_test_loss/(2*len(final_test_data)):.4f}")

    print("targets: ",t_list)
    print("predictions: ", p_list)  # get plots using Jupyter Notebook

    # print the training and validating targets for distribution verification
    testing_gamma = [v[0] for v in testing_targets]
    testing_alpha = [v[1] for v in testing_targets]
    print("testing_gamma = ", testing_gamma)
    print("testing_alpha = ", testing_alpha)


# Here we want to save our trained model
def model_save(model, path) -> None:

    torch.save(model.state_dict(), path)
    
    print("Model saved.")



if __name__ == "__main__":

    # Step 0: load graph data
    g_path = "/work/x5bai/project/Data_Files/_sim_graph_data_gnn/2025-11-11/S1_test_1111.pkl"
    print(g_path)
    with open(g_path, "rb") as file:
        data_tuple = pk.load(file)

    # Step 1: split data
    dataset = list(data_tuple.copy().values())
    train_split, val_split, test_split = 0.7, 0.15, 0.15
    final_train_data, final_test_data, final_val_data = split_data(dataset)
    print("number of training graphs = ",len(final_train_data))
    print("number of validating graphs = ",len(final_val_data))
    print("number of testing graphs = ", len(final_test_data))

    # Step 2: set up batches
    batch_train_size, batch_test_size, batch_val_size = 25, 25, 25
    train_loader = DataLoader(final_train_data, batch_size=batch_train_size, shuffle=True)
    test_loader = DataLoader(final_test_data, batch_size=batch_test_size, shuffle=True)
    val_loader = DataLoader(final_val_data, batch_size=batch_val_size, shuffle=True)

    # Step 3: training
    in_channels_model = 31
    hidden_channels_model = 4*in_channels_model
    feature_embedding_size = 2
    model_learning_rate = 0.001
    num_epoch = 500
    model = model_train(in_channels_model, hidden_channels_model, feature_embedding_size,
                model_learning_rate, num_epoch, train_loader, val_loader)

    # Step 4: testing
    model_test(model, test_loader)

    # Step 5: save model weights
    save_path = "/work/x5bai/project/Data_Files/_model_data_new/S1_test_1111.pth"
    model_save(model, save_path)

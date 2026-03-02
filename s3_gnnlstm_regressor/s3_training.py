import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import random
from itertools import combinations, permutations, product
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
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
import os
from itertools import product
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, APPNP, global_max_pool, GATv2Conv, BatchNorm

import time
import sys
sys.path.append('D:/Projects/GNN Research/Code Files/my_code_files/s3')
import models_gatlearner

def generate_triplets(data):
    triplets = []
    params = list(data.keys())  # List of parameter sets

    for param, instances in data.items(): ## 100 in total
        # Anchor and Positive from the same parameter set
        for anchor, positive in combinations(instances, 2):  ## 45 in total for each param
            # Negative from a different parameter set
            negative_param = random.choice([p for p in params if p != param])  ## randomly choose 1 from 99
            negative = random.choice(data[negative_param])  ## randomly choose 1 from 10
            triplets.append((anchor, positive, negative))
        # print(len([c for c in combinations(instances, 2)]))  ## this is a way to check it's 45 for each param

    return triplets


def generate_data_splits(train_split, test_split, val_split, triplets, data_tuple):

    if train_split + test_split + val_split == 1:

        # processing of triplet file names
        triplets_n = []
        for i in triplets:
            temp = []
            for j in i:
                temp.append(j.split('-')[0])
            triplets_n.append(tuple(temp))

        # Step 1: Create a mapping of anchor-negative pairs
        anchor_negative_map = defaultdict(list)
        k = 0
        for anchor, positive, negative in triplets_n:
            anchor_negative_map[(anchor, negative)].append((k, anchor, positive, negative))
            k += 1

        # Step 2: Get anchor-negative pairs
        anchor_negative_pairs = list(anchor_negative_map.keys())

        # Step 3: Stratified split for training, validation, and test
        train_pairs, temp_pairs = train_test_split(
            anchor_negative_pairs, test_size=1 - train_split, random_state=42
        )
        val_pairs, test_pairs = train_test_split(
            temp_pairs, test_size=val_split / (1 - train_split), random_state=42
        )

        # Step 4: Collect triplets corresponding to each split
        train_triplets = [triplet for pair in train_pairs for triplet in anchor_negative_map[pair]]
        val_triplets = [triplet for pair in val_pairs for triplet in anchor_negative_map[pair]]
        test_triplets = [triplet for pair in test_pairs for triplet in anchor_negative_map[pair]]

        train_ix = list(map(lambda x: x[0], train_triplets))
        val_ix = list(map(lambda x: x[0], val_triplets))
        test_ix = list(map(lambda x: x[0], test_triplets))

        triplets = np.array(triplets)
        train_dat = triplets[train_ix]
        test_dat = triplets[test_ix]
        val_dat = triplets[val_ix]

        final_train_data = []
        final_test_data = []
        final_val_data = []

        for i in train_dat:
            k = []
            for j in i:
                k.append(data_tuple[j])
            final_train_data.append(k)

        for i in test_dat:
            k = []
            for j in i:
                k.append(data_tuple[j])
            final_test_data.append(k)

        for i in val_dat:
            k = []
            for j in i:
                k.append(data_tuple[j])
            final_val_data.append(k)

        return final_train_data, final_test_data, final_val_data


def train(data_loader, final_val_data, device, model, criterion, optimizer, num_epochs):
    """
    Train the model using triplet loss.

    Args:
        data_loader: DataLoader that loads the triplet data.
        model: The model to be trained (e.g., GAT).
        criterion: The loss function (e.g., TripletMarginLoss).
        optimizer: The optimizer (e.g., Adam).
        num_epochs: The number of training epochs.
    """
    training_loss_list = []  # empty list to store the training loss for each epoch
    validation_loss_list = []  # empty list to store the validation loss for each epoch
    epoch_list = []  # empty list to store the ongoing epoch number

    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    best_model_state = None
    patience = 20

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0
        correct_triplets = 0
        total_triplets = 0
        margin = 1

        for anchor_data, positive_data, negative_data in data_loader:
            # Transfer the data to the appropriate device (GPU or CPU)
            anchor_data = anchor_data.to(device)
            positive_data = positive_data.to(device)
            negative_data = negative_data.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass for anchor, positive, and negative data
            anchor_out = model(anchor_data)  # Process anchor data
            positive_out = model(positive_data)  # Process positive data
            negative_out = model(negative_data)  # Process negative data

            # Calculate the triplet loss
            loss = criterion(anchor_out, positive_out, negative_out)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Update the loss summary and progress bar description
            running_loss += loss.item()
            distance_positive = F.pairwise_distance(anchor_out, positive_out, p=2)
            distance_negative = F.pairwise_distance(anchor_out, negative_out, p=2)
            correct_triplets += (distance_positive + margin < distance_negative).sum().item()
            total_triplets += len(distance_positive)

        tr_loss = running_loss / len(data_loader)
        tr_accuracy = (correct_triplets / total_triplets) * 100

        va_loss, va_accuracy = validate(final_val_data, device, model, criterion, margin=1)

        print(f"Epoch {epoch:02d} | "
              f"train loss: {tr_loss:.4f}, train accuracy: {tr_accuracy:.4f} | "
              f"val loss: {va_loss:.4f}, val accuracy: {va_accuracy:.4f}")

        training_loss_list.append(tr_loss)  # add the training loss for the current epoch to the list
        validation_loss_list.append(va_loss)  # add the training loss for the current epoch to the list
        epoch_list.append(epoch + 1)  # add the current epoch number to the list

        # Early stopping logic
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            counter = 0
            best_model_state = model.state_dict()  # Save the best model
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break


    return epoch_list, training_loss_list, validation_loss_list


def validate(data_loader, device, model, criterion, margin):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_triplets = 0
    total_triplets = 0

    with torch.no_grad():
        for anchor_data, positive_data, negative_data in data_loader:
            # Transfer data to the device
            anchor_data = anchor_data.to(device)
            positive_data = positive_data.to(device)
            negative_data = negative_data.to(device)

            # Forward pass
            anchor_out = model(anchor_data)
            positive_out = model(positive_data)
            negative_out = model(negative_data)

            # Calculate triplet loss
            loss = criterion(anchor_out, positive_out, negative_out)
            total_loss += loss.item()

            # Evaluate triplet condition
            distance_positive = F.pairwise_distance(anchor_out, positive_out, p=2)
            distance_negative = F.pairwise_distance(anchor_out, negative_out, p=2)

            correct_triplets += (distance_positive + margin < distance_negative).sum().item()
            total_triplets += len(distance_positive)

    avg_loss = total_loss / len(data_loader)
    accuracy = (correct_triplets / total_triplets) * 100

    return avg_loss, accuracy


def test(data_loader, device, model, criterion, margin):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_triplets = 0
    total_triplets = 0

    with torch.no_grad():
        for anchor_data, positive_data, negative_data in data_loader:
            # Transfer data to the device
            anchor_data = anchor_data.to(device)
            positive_data = positive_data.to(device)
            negative_data = negative_data.to(device)

            # Forward pass
            anchor_out = model(anchor_data)
            positive_out = model(positive_data)
            negative_out = model(negative_data)

            # Calculate triplet loss
            loss = criterion(anchor_out, positive_out, negative_out)
            total_loss += loss.item()

            # Evaluate triplet condition
            distance_positive = F.pairwise_distance(anchor_out, positive_out, p=2)
            distance_negative = F.pairwise_distance(anchor_out, negative_out, p=2)

            correct_triplets += (distance_positive + margin < distance_negative).sum().item()
            total_triplets += len(distance_positive)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_triplets / total_triplets * 100

    print(f"Testing Loss: {avg_loss:.4f}, Triplet Accuracy: {accuracy:.2f}%")


# Here we want to save our trained model
def model_save(model, path) -> None:
    torch.save(model.state_dict(), path)

    print("Model saved.")


class TripletDataset(Dataset):
    def __init__(self, triplets):
        """
        Args:
            triplets: A list of triplets (anchor_data, positive_data, negative_data)
        """
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_data, positive_data, negative_data = self.triplets[idx]
        anchor_data = anchor_data.to("cpu")
        positive_data = positive_data.to("cpu")
        negative_data = negative_data.to("cpu")

        return anchor_data, positive_data, negative_data


if __name__ == "__main__":

    start_time = time.time()

    sim_date = "2025-09-25"
    parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/{sim_date}/"
    iterations = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if
                  os.path.isdir(os.path.join(parent_dir, f))]
    data = {}
    records = []
    for i in iterations:
        sub_folder_name = [f for f in os.listdir(i) if os.path.isdir(os.path.join(i, f))][0].replace("=","_")
        if sub_folder_name.split("_iter_")[0] not in records:
            data[sub_folder_name.split("_iter_")[0]] = [sub_folder_name]
            records.append(sub_folder_name.split("_iter_")[0])
        else:
            data[sub_folder_name.split("_iter_")[0]] += [sub_folder_name]
    ##print(data)
    triplets = generate_triplets(data)

    # load graphs as data_tuple
    g_path = f"D:/Projects/GNN Research/Data Files/_sim_graph_data_gnn/{sim_date}/CVIS_S3.pkl"
    with open(g_path, "rb") as file:
        data_tuple = pk.load(file)

    # split data
    train_split = 0.7
    test_split = 0.15
    val_split = 0.15
    final_train_data, final_test_data, final_val_data = generate_data_splits(train_split, test_split, val_split,
                                                                             triplets, data_tuple)

    # batch setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    triplet_dataset = TripletDataset(final_train_data)
    batch_train_size = 25
    triplet_train_dataloader = DataLoader(triplet_dataset, batch_size=batch_train_size, shuffle=True)

    # model setup
    in_channels = 31
    hidden_channels = 2*in_channels
    embedding_size = 6
    model = models_gatlearner.GAT_Learner(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=embedding_size).to(device)
    model_learning_rate = 0.001
    optimizer = Adam(model.parameters(), lr=model_learning_rate)
    mixed_loss_alpha = 0.5
    mixed_loss_margin = 1
    mixed_loss_cosine = 0.2
    criterion = models_gatlearner.MixedTripletLoss(alpha=mixed_loss_alpha, margin=mixed_loss_margin, margin_cosine=mixed_loss_cosine).to(
        device)
    
    # train
    model_epochs = 500
    print(model)
    output = train(triplet_train_dataloader, final_val_data, device, model, criterion, optimizer, num_epochs=model_epochs)

    # plots
    x = output[0]
    y1 = output[1]
    y2 = output[2]

    plt.plot(x, y1, label='Training loss', linestyle='-', marker='o')
    plt.plot(x, y2, label='Validation loss', linestyle='--', marker='s')

    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss vs Epoch Number')
    plt.legend()
    plt.show()

    # test
    test(final_test_data, device, model, criterion, margin=1)

    # save
    save_path = "D:/Projects/GNN Research/Data Files/_model_data_old/CVIS_S3_minmaxtest.pth"
    model_save(model, save_path)

    end_time = time.time()
    print(f"Computing time: {(end_time - start_time) / 3600:.2f} hrs")
    print()
    
    # get min max values of ss from training data
    load_path = "D:/Projects/GNN Research/Data Files/_model_data_old/CVIS_S3_minmaxtest.pth"
    data_dict = {}
    for k, v in data_tuple.items():
        model.load_state_dict(torch.load(load_path))
        model.eval()
        with torch.no_grad():
            output = model(v.to(device))
        t = (v, output) 
        data_dict[k] = t
    ss1_list = [v[-1].tolist()[0][0] for k, v in data_dict.items()]
    ss2_list = [v[-1].tolist()[0][1] for k, v in data_dict.items()]
    ss3_list = [v[-1].tolist()[0][2] for k, v in data_dict.items()]
    ss4_list = [v[-1].tolist()[0][3] for k, v in data_dict.items()]
    ss5_list = [v[-1].tolist()[0][4] for k, v in data_dict.items()]
    ss6_list = [v[-1].tolist()[0][5] for k, v in data_dict.items()]

    min_ss1 = np.min(ss1_list)
    min_ss2 = np.min(ss2_list)
    min_ss3 = np.min(ss3_list)
    min_ss4 = np.min(ss4_list)
    min_ss5 = np.min(ss5_list)
    min_ss6 = np.min(ss6_list)
    max_ss1 = np.max(ss1_list)
    max_ss2 = np.max(ss2_list)
    max_ss3 = np.max(ss3_list)
    max_ss4 = np.max(ss4_list)
    max_ss5 = np.max(ss5_list)
    max_ss6 = np.max(ss6_list)
    print("min ss1: ", min_ss1)
    print("min ss2: ", min_ss2)
    print("min ss3: ", min_ss3)
    print("min ss4: ", min_ss4)
    print("min ss5: ", min_ss5)
    print("min ss6: ", min_ss6)
    print("max ss1: ", max_ss1)
    print("max ss2: ", max_ss2)
    print("max ss3: ", max_ss3)
    print("max ss4: ", max_ss4)
    print("max ss5: ", max_ss5)
    print("max ss6: ", max_ss6)

    values = {"min ss1": min_ss1,
              "min ss2": min_ss2,
              "min ss3": min_ss3,
              "min ss4": min_ss4,
              "min ss5": min_ss5,
              "min ss6": min_ss6,
              "max ss1": max_ss1,
              "max ss2": max_ss2,
              "max ss3": max_ss3,
              "max ss4": max_ss4,
              "max ss5": max_ss5,
              "max ss6": max_ss6,
              }

    folder_path = f"D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/{sim_date}/"
    os.makedirs(folder_path, exist_ok=True)
    final_path = os.path.join(folder_path, "CVIS_S3_minmaxss.pkl")
    with open(final_path, 'wb') as f:
        pk.dump(values, f)

    print("this number should equal the total number of simulated data: ", len(ss1_list))
    
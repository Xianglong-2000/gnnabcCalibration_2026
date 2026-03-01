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
from torch.utils.data import DataLoader as TorchDataLoader
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
from torch_geometric.loader import DataLoader as PYGDataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GATConv, GATv2Conv, BatchNorm

from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Batch
from typing import List, Tuple, Optional, Sequence, Union
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import models_gatlstmregressor


def main():

    def samples_to_sequences(file_path):  ### input = D:/Projects/GNN Research/Data Files/_sim_graph_splitted_data_gnn/2025-08-26
        folders = [os.path.join(file_path, f) for f in os.listdir(file_path) if
                   os.path.isdir(os.path.join(file_path, f))]  ### .../iteration 000/
        data_list = []
        for sample_path in folders:
            l1_folder = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if
                         os.path.isdir(os.path.join(sample_path, f))]  ### .../gamma=37.6_reg_param=1.24_iter=0/
            l2_folder = [os.path.join(l1_folder[0], f) for f in os.listdir(l1_folder[0]) if
                         os.path.isdir(os.path.join(l1_folder[0], f))]  ### .../node_feature_data/
            pkl_file_path = [os.path.join(l2_folder[0], f) for f in os.listdir(l2_folder[0])][0]  ### .../test0826_1000sims_stdnorm_log10_27f.pkl
            with open(pkl_file_path, "rb") as file:
                seq_dict = pk.load(file)  ### now the pickle is loaded

            y_tensor = None
            data = []
            indices = []
            for k,v in seq_dict.items():  ### now need to match it to the same structure as train_ds in test2()
                if hasattr(v, "y"):
                    if y_tensor is None:
                        y_tensor = v.y
                    del v.y
                data.append(v)
                idx = int(k.split("=")[-1].split(".")[0])
                indices.append(idx)
            pairs = sorted(zip(indices, data), key=lambda x: x[0])  ### making sure indices are increasing is enough and they will take data only
            ##print(pairs)  ### number should be increasing although the first number varies
            data = [comp for _, comp in pairs]  ### make sure sequence is in the right order
            ##print(idx for idx, _ in pairs)  ### it's supposed to be increasing
            data_tuple = (data, y_tensor)  ### ([Data1, ..., Data5], y) for each sample
            data_list.append(data_tuple)  ### [Sample1, Sample2, ..., Sample10] across all 1000 samples

        return data_list

    def collate_sequences_batch(batch):
        if type(batch) == tuple:
            batch = [batch]

        sequences = []
        y_seqs = []

        # Unpack outer (seq, y_seq) or just seq
        for item in batch:
            seq, y_seq = item
            sequences.append(list(seq))
            y_seqs.append(y_seq if isinstance(y_seq, torch.Tensor) else torch.as_tensor(y_seq, dtype=torch.float))

        flat_graphs: List[Data] = []
        seq_ids_: List[int] = []
        t_steps_: List[int] = []

        for seq_idx, seq in enumerate(sequences):
            for t_idx, elem in enumerate(seq):
                g, t = elem, t_idx
                flat_graphs.append(g)
                seq_ids_.append(seq_idx)
                t_steps_.append(int(t))

        flat_batch = Batch.from_data_list(flat_graphs)
        seq_ids = torch.tensor(seq_ids_, dtype=torch.long)
        t_steps = torch.tensor(t_steps_, dtype=torch.long)

        y_seq = None
        if y_seqs:
            y_seq = torch.stack(y_seqs).to(torch.float)  # [B, 2]

        return flat_batch, seq_ids, t_steps, y_seq

    def run_epoch(loader, model, optimizer, device):

        if optimizer is None:
            model.eval()
        else:
            model.train()

        total_mse = 0.0
        total_mae = 0.0
        n_samples = 0

        targets = []
        predictions = []

        for flat_batch, seq_ids, t_steps, y_seq in loader:
            # Move tensors to device
            flat_batch = flat_batch.to(device)
            seq_ids = seq_ids.to(device)
            t_steps = t_steps.to(device)
            y_seq = y_seq.to(device)  # [B, 2]
            B = y_seq.size(0)

            # Forward
            preds, lengths = model(flat_batch, seq_ids, t_steps)  # preds: [B, 2]
            loss = criterion(preds, y_seq)
            ##print("targets: ", y_seq)
            ##print("predictions: ", preds)
            ##print("num of seq sub-graphs in each sample: ", lengths)
            ##print()
            targets.append(y_seq)
            predictions.append(preds)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Metrics (sum-reduction to weight by batch size)
            with torch.no_grad():
                mse = F.mse_loss(preds, y_seq, reduction="sum").item()
                mae = F.l1_loss(preds, y_seq, reduction="sum").item()

            total_mse += mse
            total_mae += mae
            n_samples += B

        avg_mse = total_mse / (n_samples*2)
        avg_mae = total_mae / (n_samples*2)
        return avg_mse, avg_mae, targets, predictions

    def split_data(dataset: list):

        random.shuffle(dataset)

        train_size = int(len(dataset) * train_split)
        test_size = int(len(dataset) * test_split)
        val_size = int(len(dataset) * val_split)

        final_train_data = dataset[:train_size]
        final_test_data = dataset[train_size:train_size + test_size]
        final_val_data = dataset[train_size + test_size:]

        return final_train_data, final_test_data, final_val_data


    def model_save(model, path) -> None:

        torch.save(model.state_dict(), path)

        print("Model saved.")


    # main script

    dt_sim = "2025-08-20"

    file_path = f"D:/Projects/GNN Research/Data Files/_sim_graph_splitted_data_gnn/{dt_sim}"
    data_list = samples_to_sequences(file_path)
    train_split, val_split, test_split = 0.7, 0.15, 0.15
    train_ds, test_ds, val_ds = split_data(data_list)

    #torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##print("train_ds[0]: ",train_ds[0])

    b_size = 25
    train_loader = TorchDataLoader(
        train_ds, batch_size=b_size, shuffle=True,
        collate_fn=collate_sequences_batch
    )

    ##b_num = round(len(data_list)/b_size)
    ##for i,batch in zip(range(b_num),train_loader):
    ##    print(f"batch {i+1}: ", batch)

    val_loader = TorchDataLoader(
        val_ds, batch_size=b_size, shuffle=False,
        collate_fn=collate_sequences_batch
    )

    # set up the model
    num_features = 31
    num_params = 2
    model = models_gatlstmregressor.GAT_LSTM_Regressor(in_dim=num_features,
                                                 gnn_hidden_dim=2*num_features,
                                                 z_dim=2*num_features,
                                                 lstm_hidden_dim=2*num_features,
                                                 out_dim=num_params).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()  # try nn.HuberLoss() for more robustness
    print(model)

    # training
    start_time = time.time()
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    best_model_state = None

    epoch_num_list = []
    train_loss_list = []
    val_loss_list = []

    epochs = 500
    patience = 20
    for epoch in range(1, epochs + 1):
        tr_mse, tr_mae, tr_targets, tr_predictions = run_epoch(train_loader, model, optimizer, device)
        va_mse, va_mae, va_targets, va_predictions = run_epoch(val_loader, model, None, device)
        print(f"Epoch {epoch:02d} | "
              f"train MSE {tr_mse:.4f} MAE {tr_mae:.4f} | "
              f"val MSE {va_mse:.4f} MAE {va_mae:.4f}")

        epoch_num_list.append(epoch)
        train_loss_list.append(tr_mse)
        val_loss_list.append(va_mse)

        # Early stopping logic
        if va_mse < best_val_loss:
            best_val_loss = va_mse
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
    print(f"Training time: {(end_time - start_time) / 3600:.2f} hrs")

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

    # testing
    te_loader = TorchDataLoader(
        test_ds, batch_size=b_size, shuffle=False,
        collate_fn=collate_sequences_batch
    )
    te_mse, te_mae, te_targets, te_predictions = run_epoch(te_loader, model, None, device)
    print(f"test MSE {te_mse:.4f} MAE {te_mae:.4f}")
    ##print("targets: ", te_targets)  # get plots using Jupyter Notebook
    ##print("predictions: ", te_predictions)  # get plots using Jupyter Notebook

    # save model
    save_path = "D:/Projects/GNN Research/Data Files/_model_data_new/CVIS_S2.pth"
    model_save(model, save_path)

if __name__ == "__main__":
    main()
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


def test1():

    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, (hn, cn) = self.lstm(x)  # out: sequence of outputs, hn: hidden state, cn: cell state
            out = self.fc(out[:, -1, :])  # take the last
            return out, hn, cn

    # main script
    # batch_size=2, seq_len=5, input_size=2
    x = torch.randn(2, 5, 3)  # random sequence data
    y = torch.randn(2, 2)  # random target values
    print(x)
    print(y)
    input_size = 3  # one feature per time step
    hidden_size = 32
    output_size = 2  # e.g., regression
    model = SimpleLSTM(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()  # regression loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):  # train for 10 epochs
        optimizer.zero_grad()  # clear old gradients
        outputs, hn, cn = model(x)  # forward pass
        loss = criterion(outputs, y)  # compute loss
        loss.backward()  # backpropagate
        optimizer.step()  # update weights
        print(outputs)
        print(hn)
        print(cn)
        print(f'Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}')

def test2():

    class ToySeqGraphDataset(Dataset):
        def __init__(self, n_samples=200, in_channels=8, min_T=3, max_T=7, min_nodes=12, max_nodes=24):
            super().__init__()
            self.n_samples = n_samples
            self.in_channels = in_channels
            self.min_T, self.max_T = min_T, max_T
            self.min_nodes, self.max_nodes = min_nodes, max_nodes

        def __len__(self):
            return self.n_samples

        def _line_graph_edges(self, n: int) -> torch.Tensor:
            # simple chain graph: 0-1-2-...-(n-1), undirected
            src = torch.arange(0, n - 1, dtype=torch.long)
            dst = src + 1
            edges = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)  # [2, 2*(n-1)]
            return edges

        def __getitem__(self, idx: int):
            T = random.randint(self.min_T, self.max_T)
            seq: List[Data] = []
            for t in range(T):
                n = random.randint(self.min_nodes, self.max_nodes)
                x = torch.randn(n, self.in_channels)
                edge_index = self._line_graph_edges(n)
                data = Data(x=x, edge_index=edge_index)
                ##print(data)
                seq.append(data)
            # one 2-D target per sequence (your real labels go here)
            y_seq = torch.randn(2)
            ##print("seq: ", seq)  ### it shows sequences have different length
            ##print("y_seq: ", y_seq)
            return seq, y_seq

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


    class GAT_LSTM_Reg(nn.Module):
        """
        Many-to-one: one 2-D prediction per sequence of graphs.
        Pipeline:
          nodes -> GAT -> graph embedding z_t
          (z_1,...,z_T) -> LSTM -> h_last -> Linear(2)
        """
        def __init__(self, in_channels: int, gnn_hidden: int = 64, z_dim: int = 128,
                     lstm_hidden: int = 128, out_dim: int = 2):
            super().__init__()
            # --- Graph encoder to fixed-size embedding z_t ---
            self.gat1 = GATv2Conv(in_channels, gnn_hidden, heads=4, concat=True)
            self.gat2 = GATv2Conv(gnn_hidden * 4, z_dim, heads=1, concat=True)
            self.act = nn.ReLU()

            # --- Temporal encoder over sequence of embeddings ---
            self.lstm = nn.LSTM(input_size=z_dim, hidden_size=lstm_hidden, batch_first=True)

            # --- Many-to-one regression head ---
            self.head = nn.Linear(lstm_hidden, out_dim)

        @torch.no_grad()
        def _group_and_pad(self, Z: torch.Tensor, seq_ids: torch.LongTensor, t_steps: torch.LongTensor):########################################3
            """
            Z: [N_graphs, z_dim], seq_ids: [N_graphs], t_steps: [N_graphs]
            Returns:
              Z_padded: [B, T_max, z_dim]
              lengths : [B]
            """
            B = int(seq_ids.max().item()) + 1 if seq_ids.numel() > 0 else 0
            per_seq = []
            lengths = []
            for i in range(B):
                m = (seq_ids == i)
                Zi = Z[m]
                Ti = t_steps[m]
                order = torch.argsort(Ti)
                Zi = Zi[order]  # [T_i, z_dim]
                per_seq.append(Zi)
                lengths.append(Zi.size(0))
            lengths = torch.tensor(lengths, dtype=torch.long, device=Z.device)
            Z_padded = pad_sequence(per_seq, batch_first=True)  # [B, T_max, z_dim]
            return Z_padded, lengths

        def encode_graphs(self, batch: Batch) -> torch.Tensor:
            x, edge_index, b = batch.x, batch.edge_index, batch.batch
            h = self.act(self.gat1(x, edge_index))
            h = self.act(self.gat2(h, edge_index))
            Z = global_mean_pool(h, b)  # [N_graphs, z_dim]
            return Z

        def forward(self, flat_batch: Batch, seq_ids: torch.LongTensor, t_steps: torch.LongTensor):  ###########################################33
            Z = self.encode_graphs(flat_batch)  # [sum_T, z_dim]
            Z_padded, lengths = self._group_and_pad(Z, seq_ids, t_steps)  # [B, T_max, z_dim], [B]
            packed = pack_padded_sequence(Z_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hn, _) = self.lstm(packed)  # hn: [1, B, lstm_hidden]
            H_last = hn.squeeze(0)  # [B, lstm_hidden]
            preds = self.head(H_last)  # [B, 2]
            return preds, lengths

    def run_epoch(loader, model, optimizer, device):

        if optimizer is None:
            model.eval()
        else:
            model.train()

        total_mse = 0.0
        total_mae = 0.0
        n_samples = 0

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

        avg_mse = total_mse / n_samples
        avg_mae = total_mae / n_samples
        return avg_mse, avg_mae


    # main script
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 8  ### number of node features
    train_ds = ToySeqGraphDataset(n_samples=256, in_channels=in_channels)  ### there are 256 training samples
    val_ds = ToySeqGraphDataset(n_samples=64, in_channels=in_channels)  ### there are 64 validation samples
    print("train_ds[0]: ",train_ds[0])  ### show one of 256 training samples

    train_loader = TorchDataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=collate_sequences_batch
    )
    for i,batch in zip(range(8),train_loader):
        print(f"batch {i}: ", batch)

    val_loader = TorchDataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=collate_sequences_batch
    )
    #print("collate_sequences_batch(train_ds[0]): ", collate_sequences_batch(train_ds[0]))

    model = GAT_LSTM_Reg(in_channels=in_channels, gnn_hidden=64, z_dim=128,
                         lstm_hidden=128, out_dim=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()  # try nn.HuberLoss() for more robustness

    epochs = 10
    for epoch in range(1, epochs + 1):
        tr_mse, tr_mae = run_epoch(train_loader, model, optimizer, device)
        va_mse, va_mae = run_epoch(val_loader, model, None, device)
        print(f"Epoch {epoch:02d} | "
              f"train MSE {tr_mse:.4f} MAE {tr_mae:.4f} | "
              f"val MSE {va_mse:.4f} MAE {va_mae:.4f}")


def test3():
    g_path = ("D:/Projects/GNN Research/Data Files/_sim_graph_splitted_data_gnn/"
              "2025-08-26/iteration 000/gamma=37.6_reg_param=1.24_iter=0/node_feature_data/"
              "test0826_1000sims_stdnorm_log10_27f.pkl")
    print(g_path)
    with open(g_path, "rb") as file:
        data_tuple = pk.load(file)
        print(data_tuple)

    y1 = data_tuple["gamma_37.6_reg_param_1.24-Average-000_ImageNumber=30.csv"].y
    y2 = data_tuple["gamma_37.6_reg_param_1.24-Average-000_ImageNumber=60.csv"].y
    y3 = data_tuple["gamma_37.6_reg_param_1.24-Average-000_ImageNumber=90.csv"].y
    y4 = data_tuple["gamma_37.6_reg_param_1.24-Average-000_ImageNumber=110.csv"].y
    y5 = data_tuple["gamma_37.6_reg_param_1.24-Average-000_ImageNumber=130.csv"].y

    print(y1, y2, y3, y4, y5)


if __name__ == "__main__":
    test3()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, APPNP, global_max_pool, GATv2Conv, BatchNorm

from typing import List, Tuple
from torch_geometric.data import Data, Batch


# GAT Graph Regression
class GAT_Regressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_Regressor, self).__init__()

        # GAT layer on contact edges
        self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=2, concat=False)
        self.conv2 = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=2, concat=False)

        # GCN layers on lineage edges
        self.conv3 = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=2, concat=False)
        self.conv4 = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=2, concat=False)

        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):

        x, c_edge_index, l_edge_index = data.x, data.contact_edge_index, data.lineage_edge_index

        x = self.conv1(x, c_edge_index)
        x = F.elu(x)

        x = self.conv2(x, c_edge_index)
        x = F.elu(x)

        x = self.conv3(x, l_edge_index)
        x = F.elu(x)

        x = self.conv4(x, l_edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, data.batch)

        x = self.head(x)

        return x



# Attention Score Analysis for GAT Graph Regression
class GAT_Regression_Attention_Analysis(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_Regression_Attention_Analysis, self).__init__()

        # GAT layer on contact edges
        self.conv1 = GATv2Conv(in_channels = in_channels, out_channels = hidden_channels, heads = 2, concat=False)
        self.conv2 = GATv2Conv(in_channels = hidden_channels, out_channels = hidden_channels, heads = 2, concat=False)

        # GCN layers on lineage edges
        self.conv3 = GATv2Conv(in_channels = hidden_channels, out_channels = hidden_channels, heads = 2, concat=False)
        self.conv4 = GATv2Conv(in_channels = hidden_channels, out_channels = hidden_channels, heads = 2, concat=False)
        self.conv5 = GATv2Conv(in_channels = hidden_channels, out_channels = hidden_channels, heads = 2, concat=False)

        self.nn1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.nn2 = nn.Linear(hidden_channels, hidden_channels)
        self.nn3 = nn.Linear(hidden_channels, out_channels)

        self.ln1 = torch.nn.LayerNorm(hidden_channels)
        self.ln2 = torch.nn.LayerNorm(hidden_channels)
        self.ln3 = torch.nn.LayerNorm(hidden_channels)
        self.ln4 = torch.nn.LayerNorm(hidden_channels)
        self.ln5 = torch.nn.LayerNorm(hidden_channels)
        self.ln6 = torch.nn.LayerNorm(hidden_channels)
        self.ln7 = torch.nn.LayerNorm(hidden_channels)

    def forward(self, data):

        x, c_edge_index, l_edge_index = data.x, data.contact_edge_index, data.lineage_edge_index
        # x, c_edge_index = data.x, data.contact_edge_index

        x = self.conv1(x, c_edge_index)
        x = self.ln1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        #print("Layer 1 attention weights:", att1)


        x = self.conv2(x, c_edge_index)
        x = self.ln2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        #print("Layer 2 attention weights:", att2)

        x = self.conv3(x, l_edge_index)
        x = self.ln3(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        #print("Layer 2 attention weights:", att2)

        x = self.conv4(x, l_edge_index)
        x = self.ln4(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        #print("Layer 3 attention weights:", att3)

        x = self.conv5(x, l_edge_index)
        x = self.ln5(x)
        x = F.elu(x)
        #print("Layer 4 attention weights:", att4)


        # = global_mean_pool(x, data.batch)  # shape: [num_graphs, 1]
        x = torch.cat([global_mean_pool(x, data.batch),
                       global_max_pool(x, data.batch)], dim=-1)

        # MLP head for regression
        x = self.nn1(x)
        x = self.ln6(x)
        x = F.elu(x)

        x = self.nn2(x)
        x = self.ln7(x)
        x = F.elu(x)

        x = self.nn3(x)

        return x

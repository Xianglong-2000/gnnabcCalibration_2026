import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, APPNP, global_max_pool, GATv2Conv, BatchNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.data import Data, Dataset


class GAT_Learner(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_Learner, self).__init__()

        self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=2, concat=False)
        self.conv2 = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=2, concat=False)

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





class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class MixedTripletLoss(nn.Module):
    def __init__(self, alpha=0.5, margin=1.0, margin_cosine=0.2):
        super(MixedTripletLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.margin_cosine = margin_cosine

    def forward(self, anchor, positive, negative):
        triplet_loss = TripletLoss(self.margin)
        triplet_loss_cosine = TripletLossCosine(self.margin_cosine)

        loss_triplet = triplet_loss(anchor, positive, negative)
        loss_triplet_cosine = triplet_loss_cosine(anchor, positive, negative)

        mixed_loss = loss_triplet + self.alpha * loss_triplet_cosine

        return mixed_loss

class TripletLossCosine(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLossCosine, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = 1 - F.cosine_similarity(anchor, positive, dim=-1)
        distance_negative = 1 - F.cosine_similarity(anchor, negative, dim=-1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
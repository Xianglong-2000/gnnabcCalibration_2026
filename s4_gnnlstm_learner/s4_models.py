import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GATv2Conv, BatchNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class Simple_GAT_LSTM_Learner(nn.Module):

    def __init__(self, in_dim, gnn_hidden_dim, z_dim, lstm_hidden_dim, out_dim):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, gnn_hidden_dim, heads=2, concat=False)
        self.gat2 = GATv2Conv(gnn_hidden_dim, z_dim, heads=2, concat=False)
        self.lstm = nn.LSTM(input_size=z_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.head = nn.Linear(lstm_hidden_dim, out_dim)

    @torch.no_grad()
    def _group_and_pad(self, Z, seq_ids, t_steps):
        B = int(seq_ids.max().item()) + 1 if seq_ids.numel() > 0 else 0
        per_seq = []
        lengths = []
        for i in range(B):
            m = (seq_ids == i)
            Zi = Z[m]
            Ti = t_steps[m]
            order = torch.argsort(Ti)
            Zi = Zi[order]
            per_seq.append(Zi)
            lengths.append(Zi.size(0))
        lengths = torch.tensor(lengths, dtype=torch.long, device=Z.device)
        Z_padded = pad_sequence(per_seq, batch_first=True)
        return Z_padded, lengths

    def encode_graphs(self, batch):

        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        
        h = self.gat1(x, edge_index)
        h = F.elu(h)

        h = self.gat2(h, edge_index)
        h = F.elu(h)

        Z = global_mean_pool(h, b)

        return Z

    def forward(self, flat_batch, seq_ids, t_steps):

        Z = self.encode_graphs(flat_batch)

        Z_padded, lengths = self._group_and_pad(Z, seq_ids, t_steps)
        packed = pack_padded_sequence(Z_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (hn, _) = self.lstm(packed)

        H_last = hn.squeeze(0)
        
        preds = self.head(H_last)

        return preds, lengths



class GAT_LSTM_Learner(nn.Module):  #########################################################################################3
    """
    Many-to-one: one 2-D prediction per sequence of graphs.
    Pipeline:
      nodes -> GAT -> graph embedding z_t
      (z_1,...,z_T) -> LSTM -> h_last -> Linear(2)
    """

    def __init__(self, in_dim, gnn_hidden_dim, z_dim, lstm_hidden_dim, out_dim):
        super().__init__()
        # --- Graph encoder to fixed-size embedding z_t ---
        self.gat1 = GATv2Conv(in_dim, gnn_hidden_dim, heads=2, concat=False)
        self.gat2 = GATv2Conv(gnn_hidden_dim, z_dim, heads=2, concat=False)

        self.ln1 = BatchNorm(gnn_hidden_dim)
        self.ln2 = BatchNorm(gnn_hidden_dim)

        # --- Temporal encoder over sequence of embeddings ---
        self.lstm = nn.LSTM(input_size=z_dim * 2, hidden_size=lstm_hidden_dim * 2,
                            batch_first=True)  ######################### batch_first啥意思

        # --- Many-to-one regression head ---
        self.head = nn.Linear(lstm_hidden_dim * 2, out_dim)

    @torch.no_grad()
    def _group_and_pad(self, Z, seq_ids, t_steps):  ########################################3
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

    def encode_graphs(self, batch):
        x, edge_index, b = batch.x, batch.edge_index, batch.batch

        h = self.gat1(x, edge_index)
        h = self.ln1(h)
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h = self.gat2(h, edge_index)
        h = self.ln2(h)
        h = F.elu(h)

        ##Z = global_mean_pool(h, b)  # [N_graphs, z_dim]
        Z = torch.cat([global_mean_pool(h, b),
                       global_max_pool(h, b)], dim=-1)

        return Z

    def forward(self, flat_batch, seq_ids, t_steps):  ###########################################33
        Z = self.encode_graphs(flat_batch)  # [sum_T, z_dim]
        Z_padded, lengths = self._group_and_pad(Z, seq_ids, t_steps)  # [B, T_max, z_dim], [B]
        packed = pack_padded_sequence(Z_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)  # hn: [1, B, lstm_hidden]
        H_last = hn.squeeze(0)  # [B, lstm_hidden]
        preds = self.head(H_last)  # [B, 2]
        return preds, lengths





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

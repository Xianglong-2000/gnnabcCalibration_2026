import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class GAT_LSTM_Regressor(nn.Module):  #########################################################################################3
    def __init__(self, in_dim, gnn_hidden_dim, z_dim, lstm_hidden_dim, out_dim):
        super().__init__()

        self.conv1 = GATConv(in_dim, gnn_hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(gnn_hidden_dim, z_dim, heads=2, concat=False)
        self.lstm = nn.LSTM(input_size=z_dim, hidden_size=lstm_hidden_dim, batch_first=True)  ######################### batch_first啥意思
        self.head = nn.Linear(lstm_hidden_dim, out_dim)

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

        h = self.conv1(x, edge_index)
        h = F.elu(h)

        h = self.conv2(h, edge_index)
        h = F.elu(h)

        Z = global_mean_pool(h, b)  # [N_graphs, z_dim]

        return Z

    def forward(self, flat_batch, seq_ids, t_steps):  ###########################################33

        Z = self.encode_graphs(flat_batch)  # [sum_T, z_dim]

        Z_padded, lengths = self._group_and_pad(Z, seq_ids, t_steps)  # [B, T_max, z_dim], [B]
        packed = pack_padded_sequence(Z_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (hn, _) = self.lstm(packed)  # hn: [1, B, lstm_hidden]

        h_last = hn.squeeze(0)  # [B, lstm_hidden]

        predictions = self.head(h_last)  # [B, 2]

        return predictions, lengths
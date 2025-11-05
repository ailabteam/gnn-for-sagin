import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GNNLinkPredictor, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        node_u_features = x[edge_index[0]]
        node_v_features = x[edge_index[1]]
        edge_features = torch.cat([node_u_features, node_v_features], dim=1)
        predicted_edge_attr = self.mlp(edge_features)
        return predicted_edge_attr

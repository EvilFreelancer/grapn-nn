import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F


import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    """Graph Attention Network with customizable layers and activation"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, activation=torch.relu, dropout_rate=0.5, heads=1):
        super(GAT, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.convs = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate))
        for _ in range(num_layers - 2):
            # In GAT, input and output dimensions must be multiplied by the number of heads
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_rate))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate))

        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATSummarizer(torch.nn.Module):
    """Graph Attention Network for text summarization"""
    file_path = 'model_gat.pth'

    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=2):
        super(GATSummarizer, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels * heads)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels * heads)
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels * heads)
        self.classifier = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return torch.sigmoid(out)

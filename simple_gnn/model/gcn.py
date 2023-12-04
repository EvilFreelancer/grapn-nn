import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F


class GCN(torch.nn.Module):
    """Graph Convolutional Network with customizable layers and activation"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, activation=torch.relu, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, dropout=dropout_rate))
        for _ in range(num_layers - 2):  # Subtract 2 to account for the first and last layers
            self.convs.append(GCNConv(hidden_channels, hidden_channels, dropout=dropout_rate))
        self.convs.append(GCNConv(hidden_channels, out_channels, dropout=dropout_rate))

        # Edge predictor
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

    def predict_edges(self, edge_embeddings):
        """Predict the probability of a link between pairs of nodes"""
        return self.edge_predictor(edge_embeddings)


class GCNSummarizer(torch.nn.Module):
    """Graph Convolutional Network for text summarization"""
    file_path = 'model_gcn.pth'

    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(GCNSummarizer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

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

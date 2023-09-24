import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


class GCN(torch.nn.Module):
    """Graph Convolutional Network with customizable layers and activation"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, activation=torch.relu):
        super(GCN, self).__init__()

        self.activation = activation
        self.convs = torch.nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):  # Subtract 2 to account for the first and last layers
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Last layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_weight):
        for i in range(len(self.convs) - 1):  # Exclude the last layer
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.activation(x)

        # Last layer without activation
        x = self.convs[-1](x, edge_index, edge_weight)
        return x


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

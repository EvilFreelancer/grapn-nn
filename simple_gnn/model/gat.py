import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


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

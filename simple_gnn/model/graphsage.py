import torch
from torch_geometric.nn import SAGEConv, global_mean_pool
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):
    """Graph SAmple and aggreGatE"""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGESummarizer(torch.nn.Module):
    """Graph SAmple and aggreGatE for text summarization"""
    file_path = 'model_graphsage.pth'

    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(GraphSAGESummarizer, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return torch.sigmoid(out)

import torch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


class MPNNConvolutionalLayer(MessagePassing):
    """Message Passing Neural Network Convolutional Layer"""

    def __init__(self, in_channels, out_channels):
        super(MPNNConvolutionalLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Start propagating messages
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j):
        # x_j has shape [E, out_channels]
        return x_j


class MPNN(torch.nn.Module):
    """Message Passing Neural Network"""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MPNN, self).__init__()
        self.conv1 = MPNNConvolutionalLayer(in_channels, hidden_channels)
        self.conv2 = MPNNConvolutionalLayer(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class MPNNSummarizer(torch.nn.Module):
    """Message Passing Neural Network for text summarization"""
    file_path = 'model_mpnn.pth'

    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(MPNNSummarizer, self).__init__()
        self.conv1 = MPNNConvolutionalLayer(in_channels, hidden_channels)
        self.conv2 = MPNNConvolutionalLayer(hidden_channels, hidden_channels)
        self.conv3 = MPNNConvolutionalLayer(hidden_channels, hidden_channels)
        self.conv4 = MPNNConvolutionalLayer(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index, batch)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)

        return torch.sigmoid(x)

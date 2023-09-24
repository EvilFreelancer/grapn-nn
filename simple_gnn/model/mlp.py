import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self, input_dim, hidden_dim, dropout_rate=0., num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class MLPSummarizer(nn.Module):
    """Multi-Layer Perceptron for text summarization"""
    file_path = 'model_mlp.pth'

    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout_rate=0.):
        super(MLPSummarizer, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, batch):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = global_mean_pool(out, batch)
        out = self.classifier(out)
        return torch.sigmoid(out)

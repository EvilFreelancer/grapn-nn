import torch
import torch.nn as nn


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, activation=None, dropout=0.5):
        super(GraphSAGELayer, self).__init__()

        self.fc = nn.Linear(in_features * 2, out_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_matrix):
        # Aggregate neighbor features
        neighbor_features = torch.matmul(adjacency_matrix, x)

        # Concatenate node and neighbor features
        concat_features = torch.cat([x, neighbor_features], dim=1)

        # Apply the fully connected layer
        out = self.fc(concat_features)

        # Apply the activation function
        if self.activation is not None:
            out = self.activation(out)

        # Apply dropout
        out = self.dropout(out)

        return out


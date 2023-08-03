import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import json


class GraphDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph_data = self.data[idx]
        num_nodes = graph_data['num_nodes']
        edges = [item[:2] for item in graph_data['graph']]
        weights = [item[2] for item in graph_data['graph']]
        x_edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x_weights = torch.tensor(weights, dtype=torch.float)
        y = torch.tensor(graph_data['label'], dtype=torch.long)

        data = Data(edge_index=x_edges, edge_attr=x_weights, y=y)

        return data

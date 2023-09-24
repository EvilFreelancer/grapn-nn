import torch
import json
from torch_geometric.data import Data
from torch.utils.data import Dataset


class MPNNDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor(item['x'], dtype=torch.float)
        edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
        batch = torch.tensor(item['batch'], dtype=torch.long)
        y = torch.tensor(item['y'], dtype=torch.float)

        print(len(x), len(edge_index), len(batch), len(y))
        exit()

        return Data(x=x, edge_index=edge_index.t().contiguous(), batch=batch, y=y)

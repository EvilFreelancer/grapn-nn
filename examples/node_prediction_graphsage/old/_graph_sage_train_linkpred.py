# First, you need to download the dataset from the remote server
# > wget https://raw.githubusercontent.com/ZhongTr0n/JD_Analysis/main/jd_data2.json -O large_data.json

# Then create a symlink to the stable_gnn directory
# > ln -s ../../stable_gnn

import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, train_test_split_edges
from torch.optim import Adam
from simple_gnn.model import GraphSAGE
import torch.nn.functional as F

# Load dataset from file
with open('large_data.json', 'r') as f:
    graph_data = json.load(f)

# Extract list of nodes and convert it to a dictionary for fast search
node_list = [node['id'] for node in graph_data['nodes']]
node_mapping = {node_id: i for i, node_id in enumerate(node_list)}

# Create list of edges in PyTorch Geometric format
edge_index = [[node_mapping[link['source']], node_mapping[link['target']]] for link in graph_data['links']]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create an object PyTorch Geometric by Dataset
# features = torch.tensor(list(range(len(graph_data['nodes']))), dtype=torch.float)
# features = features / 1000.0
# features = features.unsqueeze(1)
features = torch.randn(len(node_list), 1)
data = Data(x=features, edge_index=edge_index)
# print(data.x)
# exit()
torch.save(data, 'data.pth')
data = train_test_split_edges(data)
data.cuda()

# Create a model object
model = GraphSAGE(data.num_node_features, 1, 1)
model.cuda()

# Обучение модели
optimizer = Adam(model.parameters(), lr=0.0001)


def train():
    model.train()
    optimizer.zero_grad()

    z = model(data.x, data.train_pos_edge_index)

    # Calculate negative and positive samples
    pos_edge = data.train_pos_edge_index.t()
    neg_edge = negative_sampling(data.train_pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge.size(0))

    pos_out = (z[pos_edge[:, 0]] * z[pos_edge[:, 1]]).sum(dim=1)
    neg_out = (z[neg_edge[:, 0]] * z[neg_edge[:, 1]]).sum(dim=1)

    #pos_loss = criterion(pos_out, torch.ones(pos_edge.size(0)).cuda())
    #neg_loss = criterion(neg_out, torch.zeros(neg_edge.size(0)).cuda())
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones(pos_out.size(0)).cuda())
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros(neg_out.size(0)).cuda())

    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss


for epoch in range(20000):
    loss = train()
    print(f"Epoch: {epoch}, Loss: {loss}")

# Save node mapping
with open('node_mapping.json', 'w') as f:
    json.dump(node_mapping, f)

# Saving model
torch.save(model.state_dict(), 'model.pth')

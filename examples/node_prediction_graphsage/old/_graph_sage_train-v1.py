# First, you need to download the dataset from the remote server
# > wget https://raw.githubusercontent.com/ZhongTr0n/JD_Analysis/main/jd_data2.json -O large_data.json

# Then create a symlink to the stable_gnn directory
# > ln -s ../../stable_gnn

import json
import torch
from torch_geometric.data import Data
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

features = torch.randn(len(node_list), 1)
labels = torch.tensor(list(range(len(graph_data['nodes']))), dtype=torch.long)

data = Data(x=features, edge_index=edge_index, y=labels)
data.cuda()

num_nodes = data.num_nodes
num_train = int(num_nodes * 0.8)  # 80% данных для обучения
num_val = int(num_nodes * 0.1)  # 10% данных для валидации
num_test = num_nodes - num_train - num_val  # Оставшиеся 10% данных для тестирования

# Создание масок
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

indices = torch.randperm(num_nodes)
train_mask[indices[:num_train]] = True
val_mask[indices[num_train:num_train + num_val]] = True
test_mask[indices[num_train + num_val:]] = True

# Добавление масок в объект данных
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# Save dataset to file
torch.save(data, 'data.pth')

# Create a model object
model = GraphSAGE(data.num_node_features, 1, len(graph_data['nodes']))
model.cuda()

# Train a model
optimizer = Adam(model.parameters(), lr=0.0001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


for epoch in range(100000):
    loss = train()
    print(f"Epoch: {epoch}, Loss: {loss}")

# Save node mapping
with open('node_mapping.json', 'w') as f:
    json.dump(node_mapping, f)

# Saving model
torch.save(model.state_dict(), 'model.pth')

import json
import torch
import numpy as np
import argparse

from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics.pairwise import cosine_similarity

from simple_gnn.model import GraphSAGE

# Avoid random
torch.manual_seed(42)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('user_json', type=str, help='JSON file with user technologies', default='user_data.json')
parser.add_argument('num_nodes', type=int, help='Number of nodes to add', default=1)
args = parser.parse_args()

# Load an object PyTorch Geometric by Dataset
data = torch.load('data.pth')
data.cuda()

# Load a model
model = GraphSAGE(data.num_node_features, 1, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()
model.cuda()

# Load node mapping
with open('node_mapping.json', 'r') as f:
    node_mapping = json.load(f)
    node_index = {index: node_id for node_id, index in node_mapping.items()}

# Loading test Graph for prediction
with open(args.user_json, 'r') as f:
    user_graph_data = json.load(f)

# Convert identifiers of nodes from user's graph to indexes of large graph
user_edge_index = []
for link in user_graph_data['links']:
    source_idx = node_mapping.get(link['source'])
    target_idx = node_mapping.get(link['target'])
    if source_idx is not None and target_idx is not None:
        user_edge_index.append([source_idx, target_idx])

user_node_index = []
for link in user_graph_data['nodes']:
    node_idx = node_mapping.get(link['id'])
    if node_idx is not None:
        user_node_index.append(node_idx)
#user_node_index.sort()

# Convert list of edges to PyTorch Geometric format
user_edge_index = torch.tensor(user_edge_index, dtype=torch.long).t().contiguous()

# Create an object PyTorch Geometric by Dataset
features = data.x
user_data = Data(x=features, edge_index=user_edge_index)
user_data.cuda()

# Generate pairs for link prediction (user node to all nodes in large graph)
user_node_indices = list(user_node_index)
all_node_indices = list(range(data.num_nodes))

# print(user_node_indices)
# print(all_node_indices)
# exit()

# Convert to tensor
user_node_indices = torch.tensor(user_node_indices, dtype=torch.long)
all_node_indices = torch.tensor(all_node_indices, dtype=torch.long)

# Get embeddings for nodes in user graph and all nodes
user_embeddings = model(data.x, user_data.edge_index)#[user_node_indices]
all_embeddings = model(data.x, data.edge_index)[all_node_indices]

# print(user_embeddings)
# print(all_embeddings)
# exit()

# Compute cosine similarities between user nodes and all nodes
similarities = cosine_similarity(all_embeddings.cpu().detach().numpy(), user_embeddings.cpu().detach().numpy())
# print(similarities[0])
# exit()

# Recommend new links
for user_node_idx in user_node_indices:
    # Find the most similar node in the large graph
    most_similar_idx = np.argmax(similarities[user_node_idx])
    most_similar_node = node_index[most_similar_idx.item()]

    print(f"{node_index[user_node_idx.item()]} >>> {most_similar_node}")

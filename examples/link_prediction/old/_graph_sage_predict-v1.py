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
# parser.add_argument('num_nodes', type=int, help='Number of nodes to add', default=1)
# parser.add_argument('max_matches', type=int, help='Amount of max matches per new node', default=10)
args = parser.parse_args()

# Load an object PyTorch Geometric by Dataset
data = torch.load('data.pth')
data.cuda()

# Load a model
model = GraphSAGE(data.num_node_features, 1, data.num_nodes)
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

user_edge_index = torch.tensor(user_edge_index, dtype=torch.long).t().contiguous()
features = data.x
user_data = Data(x=features, edge_index=user_edge_index)
user_data.cuda()

user_node_index = []
for link in user_graph_data['nodes']:
    node_idx = node_mapping.get(link['id'])
    if node_idx is not None:
        user_node_index.append(node_idx)

# Convert user's nodes to indexes of large graph
user_node_indices = torch.tensor(user_node_index, dtype=torch.long)

# Generate embeddings for all nodes in the graph
all_embeddings = model(data.x, data.edge_index)

# Generate embeddings for user's nodes
#all_embeddings = model(data.x, data.edge_index)
#user_embeddings = model(data.x, data.edge_index)
user_embeddings = model(data.x, user_data.edge_index)[user_node_indices]

# Calculate cosine similarity between user's nodes and all nodes in the graph
similarities = cosine_similarity(user_embeddings.cpu().detach().numpy(), all_embeddings.cpu().detach().numpy())

# Find the most similar nodes for each user's node
user_node_set = list(user_node_indices.numpy())
recommended_nodes = []
for index in range(len(similarities)):
    from_idx = user_node_indices[index].item()
    from_node = node_index[from_idx]

    # Determine the most similar node
    user_node_similarity = similarities[index]
    for to_idx in np.argsort(-user_node_similarity):
        # If the node is not in the user's graph
        if to_idx not in user_node_set:
            to_node = node_index[to_idx]
            recommended_nodes.append({
                'from': from_node,
                'to': to_node,
                'similarity': user_node_similarity[to_idx]
            })
            break

# Print recommended nodes
for node in recommended_nodes:
    print(node)
    # print(f"Recommended node: {node_index[node]}")

import argparse
import json
import numpy as np

import torch
from torch_geometric.data import Data

from simple_gnn.model import GraphSAGE

# Load dataset
large_dataset = torch.load('large_dataset.pth')

# Load model
model = GraphSAGE(large_dataset.num_node_features, 2, large_dataset.num_node_features)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('user_json', type=str, help='JSON file with user technologies', default='user_data.json')
args = parser.parse_args()

# Loading test Graph for prediction
with open(args.user_json, 'r') as f:
    user_graph_data = json.load(f)

# Load dataset from file
with open('large_data.json', 'r') as f:
    graph_data = json.load(f)

# Extract list of nodes and convert it to a dictionary for fast search
node_list = [node['id'] for node in graph_data['nodes']]
node_mapping = {node_id: i for i, node_id in enumerate(node_list)}
node_index = {index: node for node, index in node_mapping.items()}

# Generate subgraphs based on the small graph
user_edge_index = []
for link in user_graph_data['links']:
    source_idx = node_mapping.get(link['source'])
    target_idx = node_mapping.get(link['target'])
    # Add edge only if both nodes are on the subgraph
    if source_idx is not None and target_idx is not None:
        user_edge_index.append([source_idx, target_idx])

# Convert to PyTorch Geometric format
user_edge_index = torch.tensor(user_edge_index, dtype=torch.long).t().contiguous()

user_node_index = []
for link in user_graph_data['nodes']:
    node_idx = node_mapping.get(link['id'])
    if node_idx is not None:
        user_node_index.append(node_idx)

user_node_indices = list(user_node_index)
user_node_indices = torch.tensor(user_node_indices, dtype=torch.long)

# Create a dataset from the subgraph using the same features and labels as the original dataset
user_data = Data(x=large_dataset.x, edge_index=user_edge_index, y=large_dataset.y)

# Make prediction
prediction = model(user_data.x, user_data.edge_index)
numpy_array = prediction.cpu().detach().numpy()
sorted_indices = np.argsort(numpy_array, axis=0)
for index in sorted_indices:
    print({index[0]}, {numpy_array[index[0]][0]}, {node_index[index[0]]})

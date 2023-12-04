import argparse
import json
import numpy as np

import torch
from torch_geometric.data import Data

from simple_gnn.model import GraphSAGE, GCN, GAT

# Load dataset
large_dataset = torch.load('large_dataset.pth')

# Load model
model = GraphSAGE(large_dataset.num_node_features, 64, large_dataset.num_node_features)
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

# Convert subgraphs edges of the small graph
user_edge_index = []
for link in user_graph_data['links']:
    source_idx = node_mapping.get(link['source'])
    target_idx = node_mapping.get(link['target'])
    # Add edge only if both nodes are on the subgraph
    if source_idx is not None and target_idx is not None:
        user_edge_index.append([source_idx, target_idx])

# Convert to PyTorch Geometric format
user_edge_index = torch.tensor(user_edge_index, dtype=torch.long).t().contiguous()

# Convert subgraphs nodes of the small graph
user_node_index = []
for link in user_graph_data['nodes']:
    node_idx = node_mapping.get(link['id'])
    if node_idx is not None:
        user_node_index.append(node_idx)

# Make a mask for the subgraph nodes
user_mask = torch.zeros_like(large_dataset.x)
for idx in user_node_index:
    user_mask[idx] = 1
masked_features = large_dataset.x * user_mask

# Create a dataset from the subgraph using the same features and labels as the original dataset
user_data = Data(x=masked_features, edge_index=user_edge_index, y=large_dataset.y)


def find_neighbors(edge_index, node_idx):
    neighbors = set()
    for edge in edge_index.t().tolist():
        if edge[0] == node_idx:
            neighbors.add(edge[1])
        elif edge[1] == node_idx:
            neighbors.add(edge[0])
    return neighbors


# Вычисление вероятностей связей
def predict_edges(model, data, edge_candidates):
    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index)
        probabilities = []

        for edge in edge_candidates:
            edge_features = torch.cat([node_embeddings[edge[0]], node_embeddings[edge[1]]], dim=0)
            prob = torch.sigmoid(model.edge_predictor(edge_features.unsqueeze(0))).item()
            probabilities.append((edge, prob))

        return probabilities


# Генерация кандидатов на связи
user_existing_edges = set(tuple(sorted((e[0].item(), e[1].item()))) for e in user_edge_index.t())
user_node_pairs = set(tuple(sorted((node1, node2))) for node1 in user_node_index for node2 in user_node_index if node1 != node2)
possible_large_graph_edges = set(tuple(sorted((e[0].item(), e[1].item()))) for e in large_dataset.edge_index.t())

# Фильтрация possible_edges: только связи, которые возможны в большом графе и отсутствуют на графе пользователя
possible_edges = [list(edge) for edge in possible_large_graph_edges if edge not in user_existing_edges and edge not in user_node_pairs]

# Вычисление вероятностей связей и выбор топ-10
edge_probabilities = predict_edges(model, user_data, possible_edges)
edge_probabilities.sort(key=lambda x: x[1], reverse=True)


# Вывести топ наиболее вероятных связей
for i, (edge, prob) in enumerate(edge_probabilities[:30]):
    nodes = [node_index[edge[0]], node_index[edge[1]]]
    print(f"Связь: [{edge[0]:3}: {nodes[0]:15}] <=> [{edge[1]:3}: {nodes[1]:15}] с вероятностью {prob}")

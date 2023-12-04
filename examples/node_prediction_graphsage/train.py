import json
import torch
import random
import os

from torch_geometric.data import Data
from torch.optim import Adam, AdamW
import torch.nn.functional as F

from simple_gnn.graph.generate_subgraphs import generate_subgraphs
from simple_gnn.model import GraphSAGE, GCN, GAT
import numpy as np

torch.autograd.set_detect_anomaly(True)
np.random.seed(42)
torch.manual_seed(42)

# Load dataset from file
with open('large_data.json', 'r') as f:
    graph_data = json.load(f)

# Extract list of nodes and convert it to a dictionary for fast search
node_list = [node['id'] for node in graph_data['nodes']]
node_mapping = {node_id: i for i, node_id in enumerate(node_list)}
node_index = {index: node for node, index in node_mapping.items()}

# Create list of edges in PyTorch Geometric format
edge_index = [[node_mapping[link['source']], node_mapping[link['target']]] for link in graph_data['links']]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
features = torch.randn(len(node_list), 1)
labels = torch.tensor(list(range(len(graph_data['nodes']))), dtype=torch.long)

large_dataset = Data(x=features, edge_index=edge_index, y=labels)
torch.save(large_dataset, 'large_dataset.pth')
large_dataset.cuda()

# Load subgraphs from file, or generate them if file does not exist
if not os.path.isfile('subgraphs.json'):
    # Generate subgraphs based on the dataset
    subgraphs = generate_subgraphs(graph_data, num_subgraphs=1000, min_nodes=3, max_nodes=15)
    with open('subgraphs.json', 'w') as f:
        json.dump(subgraphs, f)
else:
    with open('subgraphs.json', 'r') as f:
        subgraphs = json.load(f)

# Generate dataset from all subgraphs
dataset = []
for i in range(len(subgraphs)):
    user_edge_index = []
    for link in subgraphs[i]['links']:
        source_idx = node_mapping.get(link['source'])
        target_idx = node_mapping.get(link['target'])
        # Add edge only if both nodes are on the subgraph
        if source_idx is not None and target_idx is not None:
            user_edge_index.append([source_idx, target_idx])
    # Convert to PyTorch Geometric format
    user_edge_index = torch.tensor(user_edge_index, dtype=torch.long).t().contiguous()

    # Convert subgraphs nodes of the small graph
    user_node_index = []
    for link in subgraphs[i]['nodes']:
        node_idx = node_mapping.get(link['id'])
        if node_idx is not None:
            user_node_index.append(node_idx)
    # Extract features of the subgraph nodes from the large graph
    user_node_indices = large_dataset.x[user_node_index]

    # Make a mask for the subgraph nodes
    user_mask = torch.zeros_like(large_dataset.x)
    for idx in user_node_index:
        user_mask[idx] = 1
    masked_features = large_dataset.x * user_mask

    # Create a dataset from the subgraph using the same features and labels as the original dataset
    user_data = Data(x=masked_features, edge_index=user_edge_index, y=labels)

    dataset.append(user_data)

# Calculate number of train, validation and test graphs
num_graphs = len(dataset)
num_train = int(num_graphs * 0.8)  # 80% for training
num_val = int(num_graphs * 0.1)  # 10% for validation
num_test = num_graphs - num_train - num_val  # Remaining 10% for testing

# Split dataset into train, validation and test sets
train_dataset = dataset
# train_dataset = dataset[:num_train]
# val_dataset = dataset[num_train:num_train + num_val]
# test_dataset = dataset[num_train + num_val:]

# Create a model object
model = GraphSAGE(large_dataset.num_node_features, 64, large_dataset.num_node_features)
model.cuda()
model.train()

# Train a model
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


# Обновление функции обучения
def train(model, optimizer, subgraph, positive_edges, negative_edges):
    model.train()
    optimizer.zero_grad()
    subgraph.cuda()

    # Получаем эмбеддинги узлов
    node_embeddings = model(subgraph.x, subgraph.edge_index)

    # Подготовка меток и объединение положительных и отрицательных примеров
    labels = torch.cat([torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))], dim=0).to(subgraph.x.device)

    # Убедимся, что edges имеет правильный тип данных
    edges = torch.cat([torch.tensor(positive_edges), torch.tensor(negative_edges)], dim=0).to(subgraph.x.device).long()

    # Функция для создания эмбеддингов рёбер
    def edge_embeddings(edges):
        return torch.cat([node_embeddings[edges[:, 0]], node_embeddings[edges[:, 1]]], dim=1)

    # Предсказание вероятности наличия связи
    predictions = torch.sigmoid(model.edge_predictor(edge_embeddings(edges))).squeeze()

    # Вычисление потерь и обновление параметров модели
    loss = F.binary_cross_entropy(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def common_neighbors(edge_index, num_nodes):
    # Создание списка соседей для каждого узла
    neighbors = {i: set() for i in range(num_nodes)}
    for edge in edge_index.t().tolist():
        neighbors[edge[0]].add(edge[1])
        neighbors[edge[1]].add(edge[0])

    return neighbors


def generate_negative_samples(edge_index, num_nodes, num_neg_samples, max_attempts=1000):
    neighbors = common_neighbors(edge_index, num_nodes)
    negative_samples = []
    attempts = 0

    while len(negative_samples) < num_neg_samples and attempts < max_attempts:
        node1 = random.choice(range(num_nodes))
        node2 = random.choice(range(num_nodes))

        # Проверяем, что узлы не связаны и имеют общих соседей
        if node1 != node2 and node2 not in neighbors[node1]:
            common_neigh = neighbors[node1].intersection(neighbors[node2])
            # Условие можно ослабить, уменьшив требуемое количество общих соседей
            if len(common_neigh) > 0:  # Узлы имеют общих соседей
                negative_samples.append([node1, node2])

        attempts += 1

    return negative_samples


# Создание положительных и отрицательных примеров
def create_edge_samples(subgraph):
    # Положительные примеры - существующие рёбра
    positive_edges = subgraph.edge_index.t().tolist()

    # Отрицательные примеры - отсутствующие рёбра
    num_neg_samples = len(positive_edges) * 2
    negative_edges = generate_negative_samples(subgraph.edge_index, subgraph.num_nodes, num_neg_samples)

    return positive_edges, negative_edges


# Обучение модели
for epoch in range(10):
    # total_loss = 0
    for subgraph in train_dataset:
        positive_edges, negative_edges = create_edge_samples(subgraph)
        if len(negative_edges) == 0:
            continue
        loss = train(model, optimizer, subgraph, positive_edges, negative_edges)
        # total_loss += loss
        print(f"Epoch: {epoch}, Loss: {loss}")

# Save model to file
torch.save(model.state_dict(), 'model.pth')

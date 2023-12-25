import torch
import random

from torch.optim import Adam
import torch.nn.functional as F

from simple_gnn.model import GraphSAGE, GCN, GAT
import numpy as np

torch.autograd.set_detect_anomaly(True)
np.random.seed(42)
torch.manual_seed(42)

from datasets import load_dataset
from torch_geometric.data import Data

# Load large graph then convert it to PyTorch Geometric format
graphs_dataset = load_dataset("evilfreelancer/jd_analysis_graphs")
large_graph = graphs_dataset["train"][1]
large_dataset = Data(
    x=torch.tensor(large_graph['node_feat'], dtype=torch.float),
    node_index={idx: node for (idx, node) in enumerate(large_graph['y'])},
    node_mapping={node: idx for (idx, node) in enumerate(large_graph['y'])},
    edge_index=torch.tensor(large_graph['edge_index'], dtype=torch.long).t().contiguous(),
    edge_attr=torch.tensor(large_graph['edge_attr'], dtype=torch.float).t().contiguous()
)
torch.save(large_dataset, 'large_dataset.pth')
large_dataset.cuda()

# Generate dataset from all subgraphs
subgraphs = graphs_dataset["jd_data2.json"]
dataset = []
for (idx, subgraph) in enumerate(subgraphs):
    dataset.append(Data(
        x=torch.tensor(subgraph['node_feat'], dtype=torch.float),
        node_index={idx: node for (idx, node) in enumerate(subgraph['y'])},
        node_mapping={node: idx for (idx, node) in enumerate(subgraph['y'])},
        edge_index=torch.tensor(subgraph['edge_index'], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(subgraph['edge_attr'], dtype=torch.float).t().contiguous()
    ))

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

# Init optimizer
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

    # Создаём эмбеддинги рёбер
    edge_embeddings = torch.cat([node_embeddings[edges[:, 0]], node_embeddings[edges[:, 1]]], dim=1)

    # Предсказание вероятности наличия связи
    predictions = torch.sigmoid(model.edge_predictor(edge_embeddings)).squeeze()

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

# Model training
loss_values = []
for epoch in range(2):
    for subgraph in train_dataset:
        positive_edges = subgraph.edge_index.t().tolist()
        negative_edges = generate_negative_samples(subgraph.edge_index, subgraph.num_nodes, len(positive_edges))
        if len(negative_edges) == 0:
            continue
        loss = train(model, optimizer, subgraph, positive_edges, negative_edges)
        loss_values.append(loss)
        print(f"Epoch: {epoch}, Loss: {loss}")

# Save model to file
torch.save(model.state_dict(), 'model.pth')

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(loss_values, label='Training Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('GAT - Training Loss Over Time')
# plt.legend()
# plt.grid(True)
# plt.savefig('training_loss.png')

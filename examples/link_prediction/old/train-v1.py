import json
import torch
import random
import os

from torch_geometric.data import Data
from torch.optim import Adam, AdamW
import torch.nn.functional as F

from simple_gnn.graph.generate_subgraphs import generate_subgraphs
from simple_gnn.model import GraphSAGE, GCN, GAT

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
    subgraphs = generate_subgraphs(graph_data, num_subgraphs=500, min_nodes=3, max_nodes=15)
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

    # Create a dataset from the subgraph using the same features and labels as the original dataset
    user_data = Data(x=large_dataset.x, edge_index=user_edge_index, y=labels)
    # user_data.cuda()

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
model = GAT(large_dataset.num_node_features, 64, large_dataset.num_node_features)
model.cuda()
model.train()

# Train a model
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

torch.autograd.set_detect_anomaly(True)

# Train a model
def train(subgraph):
    model.train()
    optimizer.zero_grad()

    subgraph.cuda()

    # Подсчет количества ребер для каждой ноды
    edge_count = {node: 0 for node in range(large_dataset.num_nodes)}
    for edge in subgraph.edge_index.t().tolist():
        # print(edge)
        edge_count[edge[0]] += 1
        edge_count[edge[1]] += 1

    # Выбор случайной ноды с одним ребром
    single_edge_nodes = [node for node, count in edge_count.items() if count == 1]
    if not single_edge_nodes:
        # Если таких нод нет, пропускаем обучение
        return None

    node_to_remove = random.choice(single_edge_nodes)

    # Узнаем ребро и его id, которое нужно удалить
    edge_to_remove = [edge for edge in subgraph.edge_index.t().tolist() if node_to_remove in edge]
    edge_to_rm_id = [idx for idx, edge in enumerate(subgraph.edge_index.t().tolist()) if node_to_remove in edge][0]

    # Исключим удаляемое ребро из графа
    edges_to_keep = [edge for edge in subgraph.edge_index.t().tolist() if node_to_remove not in edge]
    edges_to_keep = torch.tensor(edges_to_keep, dtype=torch.long).t()
    subgraph_tmp = Data(x=subgraph.x, edge_index=edges_to_keep, y=subgraph.y)
    subgraph_tmp.cuda()

    # print(edge_to_remove, edge_to_rm_id)
    # exit()

    # print(edges_to_keep.t().tolist())
    # print(subgraph.edge_index.t().tolist())
    # print(node_to_remove, edge_to_remove, edge_to_rm_id)
    # exit()

    # print((edges_to_keep != subgraph.edge_index.t().tolist()))
    # print(node_to_remove, node_index[node_to_remove])
    # exit()

    # Train a model with updated subgraph
    output_embedding = model(subgraph_tmp.x, subgraph_tmp.edge_index)
    target_embedding = model(subgraph_tmp.x, subgraph.edge_index)

    # Чуть подкорректируем векторы, чтобы они совпадали
    if output_embedding[edge_to_rm_id] != target_embedding[edge_to_rm_id]:
        print(">>>>>>>>> yes")
        #print(node_to_remove, node_index[node_to_remove], edge_to_remove, edge_to_rm_id)
        output_embedding[edge_to_rm_id] = target_embedding[edge_to_rm_id]
        #target_embedding[edge_to_rm_id] += 0.0001

    difference_mask = (output_embedding != target_embedding)
    # difference_embedding = output_embedding == target_embedding
    # print(subgraph.x, subgraph_tmp.edge_index)
    # print(difference_mask)
    # exit()

    adjusted_target = target_embedding + (0.1 * difference_mask.float())
    adjusted_output = output_embedding - (0.1 * difference_mask.float())

    # print(output_embedding, target_embedding)
    # exit()
    # Calculate loss
    loss = F.mse_loss(output_embedding, target_embedding)
    # loss = F.mse_loss(adjusted_output, adjusted_output)
    # loss = F.l1_loss(output_embedding, target_embedding)
    # loss = F.l1_loss(adjusted_output, adjusted_output)

    # Cosine similarity loss
    # cosine_similarity = F.cosine_similarity(output_embedding, target_embedding, dim=0)
    # loss = 1 - cosine_similarity.mean()

    # Binary cross entropy loss for sigmoid
    # loss = F.binary_cross_entropy_with_logits(output_embedding, target_embedding)

    loss.backward()
    optimizer.step()
    return loss


# Train a model
from tqdm import tqdm
for epoch in range(2):
    for step in range(len(train_dataset)):
        subgraph = train_dataset[step]
        min_loss = loss = train(subgraph)
        print(f"Epoch: {epoch} Step: {step} Loss: {loss}")
    #print(f"Epoch: {epoch}")

# Save model to file
torch.save(model.state_dict(), 'model.pth')

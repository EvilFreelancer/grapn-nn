import torch
from torch.optim import Adam
import numpy as np
from simple_gnn.model import GraphSAGE, GCN, GAT
from simple_gnn.negative_samples import generate_negative_samples
from simple_gnn.train import train_gnn_model
from datasets import load_dataset
from torch_geometric.data import Data

torch.autograd.set_detect_anomaly(True)
np.random.seed(42)
torch.manual_seed(42)

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

# Create a model object
model = GraphSAGE(large_dataset.num_node_features, 64, large_dataset.num_node_features)
model.cuda()
model.train()

# Init optimizer
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Model training
loss_values = []
for epoch in range(2):
    for subgraph in dataset:
        positive_edges = subgraph.edge_index.t().tolist()
        negative_edges = generate_negative_samples(subgraph.edge_index, subgraph.num_nodes, len(positive_edges))
        if len(negative_edges) == 0:
            continue
        loss = train_gnn_model(model, optimizer, subgraph, positive_edges, negative_edges)
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

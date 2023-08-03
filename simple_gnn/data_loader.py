import dgl
import torch


def load_cora_data():
    dataset = dgl.data.CoraGraphDataset()
    graph = dataset[0]

    adjacency_matrix = torch.tensor(graph.adjacency_matrix().to_dense(), dtype=torch.float)
    # Нормализация матрицы смежности
    adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdim=True)

    node_features = graph.ndata["feat"]
    labels = graph.ndata["label"]

    return adjacency_matrix, node_features, labels

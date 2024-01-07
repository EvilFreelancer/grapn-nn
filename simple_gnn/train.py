import torch
from torch.functional import F


def train_gnn_model(model, optimizer, subgraph, positive_edges, negative_edges):
    # Обновление функции обучения
    model.train()
    optimizer.zero_grad()
    # subgraph.cuda()

    # Получаем эмбеддинги узлов
    node_embeddings = model(subgraph.x, subgraph.edge_index)

    # Подготовка меток и объединение положительных и отрицательных примеров
    labels = torch.cat([torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))], dim=0).to(
        subgraph.x.device)

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

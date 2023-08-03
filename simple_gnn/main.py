import torch
from torch.optim import Adam
from graphsage import GraphSAGE


def main():
    # Загрузите данные графа и матрицу признаков вершин здесь.
    adjacency_matrix = torch.randn(10, 10)
    node_features = torch.randn(10, 5)

    input_dim = node_features.shape[1]
    hidden_dim = 32
    output_dim = 16
    depth = 2
    activation = torch.relu

    model = GraphSAGE(input_dim, hidden_dim, output_dim, depth, activation)
    optimizer = Adam(model.parameters(), lr=0.001)

    # Обучите модель с помощью обучающих данных здесь.
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(node_features, adjacency_matrix)
        # Рассчитайте функцию потерь на основе ваших задач здесь.
        loss = torch.sum(output)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


if __name__ == "__main__":
    main()

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from graphsage import GraphSAGE
from data_loader import load_cora_data


def train(model, optimizer, criterion, adjacency_matrix, node_features, labels, train_mask):
    model.train()
    optimizer.zero_grad()
    output = model(node_features, adjacency_matrix)
    loss = criterion(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    adjacency_matrix, node_features, labels = load_cora_data()

    input_dim = node_features.shape[1]
    hidden_dim = 32
    output_dim = torch.unique(labels).size(0)
    depth = 2
    activation = torch.relu

    model = GraphSAGE(input_dim, hidden_dim, output_dim, depth, activation)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    train_indices, test_indices = train_test_split(range(node_features.shape[0]), test_size=0.1, random_state=42)
    train_mask = torch.BoolTensor([i in train_indices for i in range(node_features.shape[0])])

    # Обучите модель с помощью обучающих данных здесь.
    epochs = 100
    for epoch in range(epochs):
        loss = train(model, optimizer, criterion, adjacency_matrix, node_features, labels, train_mask)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    # Сохраните обученную модель
    torch.save(model.state_dict(), "graphsage_model.pth")


if __name__ == "__main__":
    main()

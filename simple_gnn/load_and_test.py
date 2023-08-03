import torch
from graphsage import GraphSAGE
from data_loader import load_cora_data


def main():
    # Загрузка данных
    adjacency_matrix, node_features, labels = load_cora_data()

    # Загрузка обученной модели
    input_dim = node_features.shape[1]
    hidden_dim = 32
    output_dim = torch.unique(labels).size(0)
    depth = 2
    activation = torch.relu

    model = GraphSAGE(input_dim, hidden_dim, output_dim, depth, activation)
    model.load_state_dict(torch.load("graphsage_model.pth"))
    model.eval()

    # Генерация случайных входных данных
    random_node_features = torch.randn_like(node_features)

    # Передача входных данных модели и вывод результата
    output = model(random_node_features, adjacency_matrix)
    print("Output:")
    print(output)


if __name__ == "__main__":
    main()

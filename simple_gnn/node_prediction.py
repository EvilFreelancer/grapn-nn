# import torch
import torch
from graphsage import GraphSAGE
# from dgl.data.utils import split_dataset
from data_loader import load_cora_data

# Загрузите граф и создайте позитивные и негативные примеры
graph, _ = load_cora_data()
pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = train_test_split_edges(graph)

# 3. Создайте функцию потерь для задачи предсказания связей
criterion = torch.nn.BCEWithLogitsLoss()


# 4. Измените процесс обучения
def train(model, optimizer, criterion, graph, pos_train_edge, neg_train_edge):
    model.train()
    optimizer.zero_grad()

    # Вычислите векторные представления узлов
    node_embeddings = model(graph.ndata['feat'], graph.adjacency_matrix())

    # Вычислите вероятности связей для позитивных и негативных примеров
    pos_preds = compute_edge_prob(node_embeddings, pos_train_edge)
    neg_preds = compute_edge_prob(node_embeddings, neg_train_edge)

    # Вычислите функцию потерь и оптимизируйте
    loss = criterion(torch.cat([pos_preds, neg_preds]),
                     torch.cat([torch.ones(len(pos_preds)), torch.zeros(len(neg_preds))]))
    loss.backward()
    optimizer.step()

    return loss.item()


# 5. Во время тестирования модели используйте обученные векторные представления узлов
def test(model, graph, pos_test_edge, neg_test_edge):
    model.eval()

    # Вычислите векторные представления узлов
    node_embeddings = model(graph.ndata['feat'], graph.adjacency_matrix())

    # Вычислите вероятности связей для позитивных и негативных примеров
    pos_preds = compute_edge_prob(node_embeddings, pos_test_edge)
    neg_preds = compute_edge_prob(node_embeddings, neg_test_edge)

    return pos_preds, neg_preds


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def evaluate(pos_preds, neg_preds, threshold=0.5):
    preds = torch.cat([pos_preds, neg_preds]).detach().numpy()
    labels = np.concatenate([np.ones(len(pos_preds)), np.zeros(len(neg_preds))])

    # Бинаризация предсказаний
    binary_preds = (preds > threshold).astype(int)

    precision = precision_score(labels, binary_preds)
    recall = recall_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)
    roc_auc = roc_auc_score(labels, preds)

    return {"Precision": precision, "Recall": recall, "F1": f1, "ROC-AUC": roc_auc}


# 7. Обучите и протестируйте модель на вашем графе
input_dim = graph.ndata['feat'].shape[1]
hidden_dim = 64
output_dim = 64
num_layers = 2
learning_rate = 0.01
num_epochs = 100

model = GraphSAGE(input_dim, hidden_dim, output_dim, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    loss = train(model, optimizer, criterion, graph, pos_train_edge, neg_train_edge)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")

pos_preds, neg_preds = test(model, graph, pos_test_edge, neg_test_edge)
metrics = evaluate(pos_preds, neg_preds)

for metric, value in metrics.items():
    print(f"{metric}: {value}")

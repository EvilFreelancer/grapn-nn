import torch
import torch.nn.functional as F
from simple_gnn import GraphDataset, GCNSummarizer, GraphSAGESummarizer, GATSummarizer, MLPSummarizer
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 200
LEARNING_RATE = 0.001

# Load dataset
dataset = GraphDataset('graph_dataset.jsonl')
train_dataset, valid_dataset = train_test_split(dataset, test_size=0.3, random_state=2222)
train_loader = DataLoader(train_dataset, batch_size=4096)
valid_loader = DataLoader(valid_dataset, batch_size=4096)

# Initialize model and related components
model = GCNSummarizer(1200, 800).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = F.binary_cross_entropy


def train(model, dataset, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    #for data in tqdm(dataset):
    for data in dataset:
        data = data.to(device)
        optimizer.zero_grad()

        x = torch.ones(data.num_nodes, model.conv1.in_channels, device=device)
        # x = torch.ones(data.num_nodes, model.fc1.in_features, device=device) # mlp

        out = model(x, data.edge_index, data.edge_attr, data.batch)
        # out = model(x, data.batch) # mlp

        # Add an extra dimension to the target data
        loss = criterion(out, data.y.float().unsqueeze(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * 1
        total_correct += (out.round() == data.y.unsqueeze(-1)).sum().item()
        total_examples += 1

    return total_loss / total_examples, total_correct / total_examples


def validate(model, dataset, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        # for data in tqdm(dataset):
        for data in dataset:
            data = data.to(device)

            x = torch.ones(data.num_nodes, model.conv1.in_channels, device=device)
            # x = torch.ones(data.num_nodes, model.fc1.in_features, device=device) # mlp

            out = model(x, data.edge_index, data.edge_attr, data.batch)
            # out = model(x, data.batch) # mlp

            # Add an extra dimension to the target data
            loss = criterion(out, data.y.float().unsqueeze(-1))

            total_loss += loss.item()
            total_correct += (out.round() == data.y.unsqueeze(-1)).sum().item()
            total_examples += 1

    return total_loss / total_examples, total_correct / total_examples


# Train & Validate
for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
    valid_loss, valid_acc = validate(model, valid_loader, criterion, DEVICE)

    print(
        f'Epoch: {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Val Loss: {valid_loss:.5f}, Val Acc: {valid_acc:.5f}'
    )

# Saving model
torch.save(model.state_dict(), model.file_path)

import torch
from torch_geometric.data import DataLoader
from simple_gnn import MPNNSummarizer, MPNNDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Settings
in_channels = 1
hidden_channels = 8192

# Dataset
dataset = MPNNDataset('mpnn_dataset.jsonl')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model
model = MPNNSummarizer(in_channels, hidden_channels)
model = model.to(DEVICE)  # Move model to GPU if available

# Loss function
criterion = torch.nn.BCELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()  # Clear gradients
        out = model(batch.x, batch.edge_index, batch.batch)  # Forward pass
        loss = criterion(out.view(-1), batch.y.float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        total_loss += loss.item()
    print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(loader)}")

torch.save('mpnn_model.pt')

import torch
import torch.nn as nn
from test import test

class neural_network(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self.features_size = features_size
        self.layers = nn.Sequential(
                nn.Linear(self.features_size, 16),
                nn.ReLU(),
                nn.Linear(16,32),
                nn.ReLU(),
                nn.Linear(32,1)
                )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def train_model(model, dataloader, test_dataloader, epochs, lr=0.01):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch} ------------\n")
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        for i, (x,y) in enumerate(dataloader):
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i%100 == 0:
                print("Batch:",i+1, "Loss: ", loss.item())
        test(model, test_dataloader)
    torch.save(model.state_dict(), "models/model.pth")
    print("Model sucessfully saved to the models directory")

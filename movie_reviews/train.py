import torch
from torch import nn
from test import test
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} to train the model")

class neural_network(nn.Module):
    def __init__(self, vocab_size=10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.layers = nn.Sequential(
                nn.Linear(vocab_size, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
                )
    def forward(self,x):
        logits = self.layers(x)
        return logits

model = neural_network()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(model, dataloader, epochs, test_dataloader):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    model = model.to(device)
    model.train()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    for epoch in range(epochs):
        train_loss = 0
        for i, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # compute prediction error
            logits = model(X)
            # print(logits)
            loss = loss_fn(logits, y)
            train_loss += loss
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%100 == 0:
                loss, current = loss.item(), i+1
                print(f"loss: {loss:>7f}, current epoch is {current}")
        train_loss /= num_batches 
        print("-"*5,"\n",f"Train Average Loss is {train_loss:>8f} --- epoch: {epoch+1}", "\n", "-"*5)
        test(model, test_dataloader)

    torch.save(model.state_dict(), "models/model.pth")
    print("Saved the model in models directory with name 'model.pth' ")

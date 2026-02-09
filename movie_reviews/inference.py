import torch
from train import neural_network

model = neural_network()
model.load_state_dict(torch.load("models/model.pth", weights_only = True))
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def run(model, test_data, device):
    model = model.to(device)
    model.eval()

    test_data = test_data.to(device)

    logits = model(test_data)

    predicted = logits.argmax(1)

    return predicted


def infer(test_data):
    preds = run(model, test_data, device)
    print(preds)
    return preds    

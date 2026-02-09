import torch
from train import neural_network

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = neural_network(8).to(device)
model.load_state_dict(torch.load("models/model.pth", weights_only = True))

def inference(data):
    data = data.to(device)
    model.eval()
    preds = model(data) 
    return preds

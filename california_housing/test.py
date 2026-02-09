import torch

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
loss_fn = torch.nn.L1Loss(reduction="mean")

def test(model, dataloader):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    model = model.to(device) 
    model.eval()
    test_loss = 0
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds.view(-1), y)
        test_loss += loss.item()
    print("Average Test Loss : ", test_loss/num_batches)

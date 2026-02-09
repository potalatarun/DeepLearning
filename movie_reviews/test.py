import torch

loss_fn = torch.nn.CrossEntropyLoss()
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
def test(model, dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model = model.to(device)
    model.eval() 
    loss ,correct = 0, 0
    with torch.no_grad():
        for i, (text ,label) in enumerate(dataloader):
            text, label = text.to(device), label.to(device)
            logits = model(text)
            loss += loss_fn(logits, label)
            correct += (logits.argmax(1)==label).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"Avg Test Loss is: {loss:>8f}, Accuracy is {correct*100:>0.1f}")



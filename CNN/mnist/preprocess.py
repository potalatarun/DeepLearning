import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from datasets import load_from_disk
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

class mnist_dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = Image.fromarray(np.array(self.images[idx]))
        np_array = np.array(img)
        # image = ToTensor(np_array)
        label = self.labels[idx]
        return (
                torch.tensor(np_array,dtype=torch.float),
                torch.tensor(label, dtype=torch.long)
                )

data = load_from_disk("datasets")
train_data = data['train']
test_data = data['test']

train_dataset = mnist_dataset(train_data['image'], train_data['label'])
test_dataset = mnist_dataset(test_data['image'], test_data['label'])

train_dataloader = DataLoader(dataset = train_dataset, batch_size = 640, shuffle = True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 640, shuffle = True)

for images, labels in test_dataloader:
    print(images.shape, labels.shape)
    break

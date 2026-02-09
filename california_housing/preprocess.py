import pandas as pd
import torch
from torch.utils.data import Dataset

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_target = train['MedHouseVal']
train_data = train.drop(columns=['MedHouseVal'])

test_target = test['MedHouseVal']
test_data = test.drop(columns=['MedHouseVal'])

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


class california_housing(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return self.features.shape[0] 

    def __getitem__(self, idx):
        data = self.features.iloc[idx]
        labels = self.targets.iloc[idx]

        return (
                torch.tensor(data, dtype = torch.float),
                torch.tensor(labels, dtype = torch.float)
                )

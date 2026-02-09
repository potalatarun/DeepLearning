import torch
import pandas as pd
from preprocess import california_housing
from torch.utils.data import DataLoader
from train import neural_network
from train import train_model
from inference import inference

# collect data
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_data = train.drop(columns=['MedHouseVal'])
train_target = train['MedHouseVal']

test_data = test.drop(columns=['MedHouseVal'])
test_target = test['MedHouseVal']

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

#train_target = train_target / 100000
#test_target = test_target / 100000

train_dataset = california_housing(train_data, train_target)
test_dataset = california_housing(test_data, test_target)

train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = 64,
        shuffle = True
        )
test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = True
        )


# train step
# model = neural_network(8)
# train_model(model, train_dataloader, test_dataloader, 10)
j = 3
for i, (x,y) in enumerate(test_dataloader):
    preds = inference(x)
    print(preds.item())
    print(y)
    if i==j:
        break


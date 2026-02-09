from preprocess import build_vocab
import pandas as pd
import torch 
from torch.utils.data import DataLoader
from preprocess import IMDB_Dataset 
from train import train
from train import neural_network
from inference import infer

train_data  = pd.read_csv("datasets/train.csv")
test_data   = pd.read_csv("datasets/test.csv")

word_index = build_vocab(train_data['text'])

imdb_train_dataset = IMDB_Dataset(texts = train_data['text'], labels = train_data['label'], word_index = word_index, max_len = 1000)
imdb_test_dataset = IMDB_Dataset(texts = test_data['text'], labels = test_data['label'], word_index = word_index, max_len = 1000)

imdb_train_dataloader = DataLoader(
        dataset = imdb_train_dataset,
        batch_size = 150,
        shuffle = True
        )

imdb_test_dataloader = DataLoader(
        dataset = imdb_test_dataset,
        batch_size = 32,
        shuffle = True
        )

#model = neural_network()
#train(model, imdb_train_dataloader, 10, imdb_test_dataloader)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
for text, label in imdb_test_dataloader:
    texts = text
    labels = label
    print(len(labels))
    preds = infer(texts)
    print("Label : ", labels)
    labels = labels.to(device)
    print((preds == labels).sum(), len(labels))
    break

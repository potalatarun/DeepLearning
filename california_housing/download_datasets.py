import datasets
from datasets import load_dataset


dataset = load_dataset("gvlassis/california_housing")

train = dataset['train'].to_csv("datasets/train.csv")
test = dataset['test'].to_csv("datasets/test.csv")

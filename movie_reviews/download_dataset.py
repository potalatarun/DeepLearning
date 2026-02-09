import datasets
from datasets import load_dataset


imdb_dataset = load_dataset('imdb')


#imdb_dataset.save_to_disk("./datsets")
train = imdb_dataset['train']
test = imdb_dataset['test']

train.to_csv('datasets/test.csv', index=False)
test.to_csv("datasets/train.csv", index=False)
print(train.shape, test.shape)

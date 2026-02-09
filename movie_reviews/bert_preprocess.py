import pandas as pd
from transformers import AutoTokenizer


# load train and test data
train_data = pd.read_csv("datasets/train.csv")
test_data = pd.read_csv("datasets/test.csv")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
word_index = tokenizer.get_vocab()

def tokenize_dataset(data, word_index):
    # we need to tokenize the raw data 
    # here tokenize means we need to convert the raw text into numerical values

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # word_index = tokenizer.get_vocab()

    processed_data = []

    new_word = len(word_index)
    for row in data['text']:
        temp = []
        for word in row:
            if word in word_index:
                temp.append(word_index[word])
            else:
                temp.append(new_word)
                word_index[word] = new_word
                new_word += 1
        processed_data.append(temp)
    return processed_data, word_index

train_processed, word_index = tokenize_dataset(train_data, word_index)
test_processed, word_index = tokenize_dataset(test_data, word_index)



train_processed = pd.DataFrame(
        {
            "text" : train_processed,
            "label" : train_data['label']
            }
        )
test_processed = pd.DataFrame(
        {
            "text" : test_processed,
            "label" : test_data['label']
            }
        )
train_processed.to_csv("datasets/processed/train.csv", index=False)
test_processed.to_csv("datasets/processed/test.csv", index=False)

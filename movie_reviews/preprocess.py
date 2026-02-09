from collections import Counter
import re 
import torch
from torch.utils.data import Dataset

# text pre-prorcessing
def clean_text(text):
    """
    this function cleans the given text.
    removes the characters like html <> something like these
    """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


# Build Vocabulary
def build_vocab(texts, max_vocab_size = 10000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_vocab_size - 2)

    word_index = {
            "<PAD>" : 0,
            "<UNK>" : 1,
            }

    for i, (word, _) in enumerate(most_common, start=2):
        word_index[word] = i
    return word_index


# Text -> integer 
def text_to_sequence(text, word_index):
    return [
            word_index.get(word, word_index["<UNK>"]) for word in text.split()
            ]


# Padding 
def pad_sequence(seq, max_len):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))


# vectorize sequence
def vectorize_sequence(seq, vocab_size):
    vec = torch.zeros(vocab_size)
    for idx in seq:
        if idx < vocab_size:
            vec[idx] = 1.0
    return vec



# PyTorch Dataset

class IMDB_Dataset(Dataset):
    def __init__(self, texts, labels, word_index, max_len):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        text = clean_text(text) 
        seq = text_to_sequence(text, self.word_index)
        seq = pad_sequence(seq, self.max_len)
        seq = vectorize_sequence(seq, 10000)

        return (
                torch.tensor(seq, dtype=torch.float),
                torch.tensor(label, dtype = torch.long)
                )

#! coding utf-8

from huggingface_hub import list_datasets
from datasets import load_dataset
from pprint import pprint
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


def test():
    all_datasets = list_datasets()

    emotions = load_dataset("emotion")

    train_ds = emotions['train']
    # pprint(train_ds[:5])

    emotions.set_format(type='pandas')
    df = emotions['train'][:]
    # print(df.head())

    def label_in2str(row):
        return emotions['train'].features['label'].int2str(row)

    df['label_name'] = df['label'].apply(label_in2str)
    # print(df.head())

    df['label_name'].value_counts(ascending=True).plot.barh()
    plt.show()


def tokenization_test():
    text = "Tokenizing text is a core task of NLP."
    tokenized_text = list(text)
    token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}

    input_ids = [token2idx[token] for token in tokenized_text]

    input_ids = torch.tensor(input_ids)
    one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
    print(one_hot_encodings.shape)


if __name__ == "__main__":
    tokenization_test()
    pass

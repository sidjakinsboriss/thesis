from collections import Counter

import numpy as np
import pandas
import pandas as pd
import torch
from gensim.models import KeyedVectors
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib import pyplot as plt
from nltk import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

wv = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)
padding_token = '<pad>'
padding_vector = np.zeros(wv.vector_size)

vocab_size = len(wv.key_to_index) + 1
new_vectors = np.zeros((vocab_size, wv.vector_size))
new_vectors[0] = padding_vector
new_vectors[1:] = wv.vectors

new_wv = KeyedVectors(wv.vector_size)
new_wv.add_vectors([padding_token] + wv.index_to_key, new_vectors)


class PadSequence:
    def __call__(self, batch):
        data = [item['indices'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])

        # Pad the sequences
        padded_data = pad_sequence(data, batch_first=True)

        return padded_data, labels


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [new_wv.key_to_index.get(word) for word in word_tokenize(text)]
        indices = torch.tensor([idx if idx is not None else 0 for idx in indices], dtype=torch.int32)

        label = self.labels[idx]
        return {'indices': indices[:100], 'label': label}


class DatasetSplitHandler:
    def __init__(self, df: pandas.DataFrame):
        self.df = df
        self.mlb = MultiLabelBinarizer()
        self.tag_distribution = df['TAGS'].value_counts(normalize=True)
        self.indices = []
        self.weights = torch.FloatTensor(new_wv.vectors)
        self.email_train = None
        self.email_test = None
        self.email_val = None
        self.tag_train = None
        self.tag_test = None
        self.tag_val = None

    def split_dataset(self):
        # Ignore combination of labels that occur less than 10 times
        # label_counts = self.df['TAGS'].value_counts()
        # selected_labels = label_counts[label_counts >= 10].index
        # subset_df = self.df[self.df['TAGS'].isin(selected_labels)]

        X = np.zeros(len(self.df))
        y = self.df.iloc[:, 2:]

        n_splits = 10
        skf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

        # Perform stratified K-Fold splitting
        for i, (_, test_index) in enumerate(skf.split(X, y)):
            self.indices.append(test_index)

    def encode_labels(self):
        """
        Encode tags into series of 0s and 1s
        """
        self.df['TAGS'] = self.df['TAGS'].str.split(', ')
        encoded = self.mlb.fit_transform(self.df['TAGS'])
        self.df[self.mlb.classes_] = encoded

    def plot_tag_distribution(self):
        self.tag_distribution.plot(
            kind='bar',
            figsize=(10, 4),
            title='Tag Distribution',
            ylabel='Proportion of observations'
        )
        plt.show()

    def plot_split_tag_distribution(self):
        split = pd.DataFrame({
            'tag_train': Counter(self.tag_train),
            'tag_val': Counter(self.tag_val),
            'tag_test': Counter(self.tag_test)
        }).reindex(self.tag_distribution.index)

        split = split / split.sum(axis=0)
        split.plot(
            kind='bar',
            figsize=(10, 4),
            title='Tag Distribution per Split',
            ylabel='Proportion of observations'

        )

        plt.show()

    def get_data_loaders(self):
        """
        Yields data loaders for training, testing, and validation sets.
        """
        for i in range(10):
            indices = [num for num in [k for k in range(10)] if num not in [i, (i + 1) % 10]]

            test_indices = self.indices[i]
            val_indices = self.indices[(i + 1) % 10]
            train_indices = [self.indices[k] for k in indices]
            train_indices = [idx for sublist in train_indices for idx in sublist]

            email_test = self.df['CONTENT'].iloc[test_indices]
            tag_test = self.df[self.mlb.classes_].iloc[test_indices]

            email_val = self.df['CONTENT'].iloc[val_indices]
            tag_val = self.df[self.mlb.classes_].iloc[val_indices]

            email_train = self.df['CONTENT'].iloc[train_indices]
            tag_train = self.df[self.mlb.classes_].iloc[train_indices]

            train_tags = torch.Tensor(tag_train.values)
            val_tags = torch.Tensor(tag_val.values)
            test_tags = torch.Tensor(tag_test.values)

            train_dataset = TextDataset(list(email_train), train_tags)
            val_dataset = TextDataset(list(email_val), val_tags)
            test_dataset = TextDataset(list(email_test), test_tags)

            train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=PadSequence())
            val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=PadSequence())
            test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=PadSequence())

            yield train_loader, val_loader, test_loader

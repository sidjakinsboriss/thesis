from collections import Counter

import pandas
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import TensorDataset, DataLoader


class DatasetSplitHandler:
    def __init__(self, df: pandas.DataFrame):
        self.df = df
        self.mlb = MultiLabelBinarizer()
        self.tag_distribution = df['TAGS'].value_counts(normalize=True)
        self.email_train = None
        self.email_test = None
        self.email_val = None
        self.tag_train = None
        self.tag_test = None
        self.tag_val = None

    def split_dataset(self):
        # Ignore combination of labels that occur less than 10 times
        label_counts = self.df['TAGS'].value_counts()
        selected_labels = label_counts[label_counts >= 10].index
        subset_df = self.df[self.df['TAGS'].isin(selected_labels)]
        tags = subset_df.iloc[:, 2:]

        self.email_train, email_remaining, self.tag_train, tag_remaining = train_test_split(
            subset_df['CONTENT'], tags, train_size=0.8, stratify=tags, random_state=42
        )
        self.email_val, self.email_test, self.tag_val, self.tag_test = train_test_split(
            email_remaining, tag_remaining, train_size=0.5, stratify=tag_remaining, random_state=42
        )

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
        # Convert preprocessed data into PyTorch tensors
        train_vectors = torch.Tensor(self.email_train)
        train_tags = torch.Tensor(self.tag_train.values)
        val_vectors = torch.Tensor(list(self.email_val))
        val_tags = torch.Tensor(self.tag_val.values)
        test_vectors = torch.Tensor(list(self.email_test))
        test_tags = torch.Tensor(self.tag_test.values)

        # Create data loaders
        train_dataset = TensorDataset(train_vectors, train_tags)
        val_dataset = TensorDataset(val_vectors, val_tags)
        test_dataset = TensorDataset(test_vectors, test_tags)

        train_loader = DataLoader(train_dataset, batch_size=100)
        val_loader = DataLoader(val_dataset, batch_size=100)
        test_loader = DataLoader(test_dataset, batch_size=100)

        return train_loader, val_loader, test_loader

import random

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer


class DatasetHandler:
    def __init__(self, df: pd.DataFrame):
        """
        This class handles various dataset operations, such as encoding labels,
        splitting the dataset, and under-sampling.
        """
        self.df = df
        self.mlb = MultiLabelBinarizer()
        self.tag_distribution = df['TAGS'].value_counts(normalize=True)
        self.indices = []

    def split_dataset(self):
        """
        Splits the dataset into 10 subsets stratified by tags
        """
        X = np.zeros(len(self.df))
        y = self.df.iloc[:, 4:]

        n_splits = 10
        skf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=42,
                                        shuffle=True)

        for _, test_index in skf.split(X, y):
            self.indices.append(test_index)

    def are_unique_splits(self):
        """
        Check if MultilabelStratifiedKFold results in non-overlapping splits of the dataset
        """
        splits = self.indices
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                intersection = set(splits[i]) & set(splits[j])
                if len(intersection) > 0:
                    return False
        return True

    def encode_labels(self):
        """
        Multi-hot encoding of labels
        """
        self.df['TAGS'] = self.df['TAGS'].str.split(', ')
        encoded = self.mlb.fit_transform(self.df['TAGS'])
        self.df[self.mlb.classes_] = encoded

    def under_sample(self, train_indices):
        """
        Under samples the emails with 'not-ak' label
        @param train_indices: indices used for the training set
        """
        ak_indices, not_ak_indices = [], []
        for idx in train_indices:
            if self.df.iloc[idx]['not-ak']:
                not_ak_indices.append(idx)
            else:
                ak_indices.append(idx)

        train_indices = ak_indices + random.sample(not_ak_indices, 200)
        random.shuffle(train_indices)

        return np.array(train_indices)

    def get_indices_for_optimization(self, under_sample=False):
        """
        Returns indices for training, testing, and validation sets.

        @param under_sample: whether to under-sample emails with 'not-ak' label
        @return: indices for training, testing, and validation sets
        """
        indices = [num for num in [k for k in range(10)] if num not in [0, 1]]

        test_indices = self.indices[0]
        val_indices = self.indices[1]
        train_indices = [self.indices[k] for k in indices]
        train_indices = [idx for sublist in train_indices for idx in
                         sublist]
        if under_sample:
            train_indices = self.under_sample(train_indices)
        return train_indices, val_indices, test_indices

    def get_indices(self, under_sample=False):
        """
        Returns indices for training, testing, and validation sets.

        @param under_sample: whether to under-sample emails with 'not-ak' label
        @return: indices for training, testing, and validation sets
        """
        for i in range(10):
            indices = [num for num in [k for k in range(10)] if
                       num not in [i, (i + 1) % 10]]

            test_indices = self.indices[i]
            val_indices = self.indices[(i + 1) % 10]
            train_indices = [self.indices[k] for k in indices]
            train_indices = [idx for sublist in train_indices for idx in
                             sublist]
            if under_sample:
                train_indices = self.under_sample(train_indices)

            yield train_indices, val_indices, test_indices

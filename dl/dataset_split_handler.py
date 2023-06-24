import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


class DatasetHandler:
    def __init__(self, df: pandas.DataFrame, word_embeddings, include_parent_email):
        self.df = df
        self.mlb = MultiLabelBinarizer()
        self.tag_distribution = df['TAGS'].value_counts(normalize=True)
        self.indices = []
        self.word_embeddings = word_embeddings
        self.include_parent_email = include_parent_email
        self.email_train = None
        self.email_test = None
        self.email_val = None
        self.tag_train = None
        self.tag_test = None
        self.tag_val = None

    def split_dataset(self):
        """
        Splits the dataset into 10 subsets stratified by tags
        """
        X = np.zeros(len(self.df))
        y = self.df.iloc[:, 5:]

        n_splits = 10
        skf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

        for _, test_index in skf.split(X, y):
            self.indices.append(test_index)

        if os.path.exists('list.pickle'):
            with open('list.pickle', 'rb') as fp:
                self.indices = pickle.load(fp)
        else:
            for _, test_index in skf.split(X, y):
                self.indices.append(test_index)

            # Save the splits
            with open('list.pickle', 'wb') as fp:
                pickle.dump(self.indices, fp)

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

    def remove_not_ak_tags(self, train_indices):
        tag_train = self.df[self.mlb.classes_].iloc[train_indices]
        tags = self.df[self.mlb.classes_]

        counts = []

        for col in self.mlb.classes_:
            positive_samples = tag_train[tag_train[col] == 1].shape[0]
            counts.append(positive_samples)

        min_count = min(counts)

        not_ak_indices = []
        ak_indices = []

        for idx in train_indices:
            if tags.iloc[idx].tolist() == [0, 1, 0, 0, 0]:
                not_ak_indices.append(idx)
            else:
                ak_indices.append(idx)

        not_ak_indices = not_ak_indices[:min_count]
        train_indices = not_ak_indices + ak_indices
        random.shuffle(train_indices)

        return train_indices

    def get_balanced_training_dataset(self, train_indices):
        tag_train = self.df[self.mlb.classes_].iloc[train_indices]
        tags = self.df[self.mlb.classes_]

        counts = []
        class_counts = [0 for _ in range(5)]

        for col in self.mlb.classes_:
            positive_samples = tag_train[tag_train[col] == 1].shape[0]
            counts.append(positive_samples)

        min_count = min(counts)

        included_indices = []

        for idx in train_indices:
            tag = tags.iloc[idx].tolist()
            pos_indices = [i for i in range(len(tag)) if tag[i] == 1]

            include_idx = True
            for i in pos_indices:
                include_idx = include_idx and class_counts[i] < min_count

            if include_idx:
                included_indices.append(idx)
                for i in pos_indices:
                    class_counts[i] += 1

        return included_indices

    def add_parent_email(self, train_indices) -> pandas.Series:
        emails_with_parent_email = []

        for idx in train_indices:
            email_info = self.df[['CONTENT', 'ID', 'PARENT_ID']].iloc[idx]
            email = email_info['CONTENT']
            parent_email = self.df.loc[self.df['ID'] == email_info['PARENT_ID'], ['CONTENT']]

            if parent_email.empty:
                parent_email = None
            else:
                parent_email = parent_email.iat[0, 0]

            emails_with_parent_email.append([parent_email, email])

        return pandas.Series(emails_with_parent_email)

    def get_indices(self):
        for i in range(10):
            indices = [num for num in [k for k in range(10)] if num not in [i, (i + 1) % 10]]

            test_indices = self.indices[i]
            val_indices = self.indices[(i + 1) % 10]
            train_indices = [self.indices[k] for k in indices]
            train_indices = [idx for sublist in train_indices for idx in sublist]

            return train_indices, val_indices, test_indices

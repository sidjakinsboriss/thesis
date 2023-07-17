import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


class DatasetHandler:
    def __init__(self, df: pd.DataFrame):
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
        skf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

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

    def add_powerlabels(self):
        self.df['POWERLABEL'] = self.df.apply(
            lambda x: 16 * x['existence'] + 8 * x['not-ak'] + 4 * x['process'] + 2 * x['property'] + 1 * x[
                'technology'],
            axis=1)

    def plot_tag_distribution(self):
        self.tag_distribution.plot(
            kind='bar',
            figsize=(10, 4),
            title='Tag Distribution',
            ylabel='Proportion of observations'
        )
        plt.show()

    def oversampled_df(self, train_indices):
        train_df = self.df.iloc[train_indices, :]
        powercount = {}
        powerlabels = np.unique(train_df['POWERLABEL'])
        for p in powerlabels:
            powercount[p] = np.count_nonzero(train_df['POWERLABEL'] == p)

        maxcount = np.max(list(powercount.values()))
        for p in powerlabels:
            gapnum = maxcount - powercount[p]
            temp_df = train_df.iloc[np.random.choice(np.where(train_df['POWERLABEL'] == p)[0], size=gapnum)]
            train_df = train_df.append(temp_df)

        indices = train_df.index.values
        return indices

    def get_parent_indices(self, indices):
        parent_indices = []
        for idx in indices:
            parent_id = self.df.iloc[idx]['PARENT_ID']
            parent_idx = self.df.index[self.df['ID'] == parent_id].tolist()

            if parent_idx:
                parent_indices.append(parent_idx[0])
            else:
                parent_indices.append(idx)

        return np.array(parent_indices)

    @staticmethod
    def concatenate_with_parent_indices(sequences, indices, parent_sequences, parent_indices):
        res = []

        for i in range(len(sequences)):
            if indices[i] != parent_indices[i]:
                res.append(np.concatenate((parent_sequences[i], sequences[i])))
            else:
                res.append(np.concatenate((np.zeros(100), sequences[i])))

        return np.array(res, dtype=int)

    def get_indices(self):
        for i in range(10):
            indices = [num for num in [k for k in range(10)] if num not in [i, (i + 1) % 10]]

            test_indices = self.indices[i]
            val_indices = self.indices[(i + 1) % 10]
            train_indices = [self.indices[k] for k in indices]
            train_indices = [idx for sublist in train_indices for idx in sublist]

            yield train_indices, val_indices, test_indices

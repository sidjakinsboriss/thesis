import os
from collections import Counter

import pandas
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import TensorDataset


def encode_labels(df: pandas.DataFrame):
    tags = df['TAGS'].tolist()
    mlb = MultiLabelBinarizer()
    tags = mlb.fit_transform(tags)
    df[mlb.classes_] = tags

    tags = df["TAGS"].tolist()
    X_train, X_, y_train, y_ = train_test_split(
        df['CONTENT'], tags, train_size=0.8, stratify=tags, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_, random_state=42
    )

    tag_dis = df['TAGS'].value_counts(normalize=True)
    split = pd.DataFrame({
        'y_train': Counter(y_train),
        'y_val': Counter(y_val),
        'y_test': Counter(y_test)
    }).reindex(tag_dis.index)

    split = split / split.sum(axis=0)
    split.plot(
        kind='bar',
        figsize=(10, 4),
        title='Tag Distribution per Split',
        ylabel='Proportion of observations'
    )
    plt.show()
    print("YO")


def split_dataset(df: pandas.DataFrame):
    """
    Split the dataset into training, testing, and validation sets
    """
    tags = df["TAGS"].tolist()
    X_train, X_, y_train, y_ = train_test_split(
        df['CONTENT'], tags, train_size=0.8, stratify=tags, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_, random_state=42
    )

    split = pd.DataFrame({
        'y_train': Counter(', '.join(i for i in row) for row in mlb.inverse_transform(y_train)),
        'y_val': Counter(', '.join(i for i in row) for row in mlb.inverse_transform(y_val)),
        'y_test': Counter(', '.join(i for i in row) for row in mlb.inverse_transform(y_test))
    }).reindex(tag_dis.index)

    split = split / split.sum(axis=0)
    split.plot(
        kind='bar',
        figsize=(10, 4),
        title='Tag Distribution per Split',
        ylabel='Proportion of observations'
    )
    plt.show()
    print("YO")


if __name__ == "__main__":
    df = pandas.read_json(os.path.join(os.getcwd(), "../data/preprocessed.json"), orient='index')
    encode_labels(df)  # PyTorch only accepts digits
    tag_dis = df['TAGS'].value_counts(normalize=True)

    # Plot tag distribution
    tag_dis.plot(
        kind='bar',
        figsize=(10, 4),
        title='Tag Distribution of All Observations',
        ylabel='Proportion of observations'
    )
    # plt.show()

    # Split the dataset
    train_df, test_df, val_df = split_dataset(df)

    # Convert preprocessed data into PyTorch tensors
    train_vectors = torch.Tensor(train_df["CONTENT"])
    train_labels = torch.Tensor(train_df["TAGS"])
    val_vectors = torch.Tensor(val_df["CONTENT"])
    val_labels = torch.Tensor(val_df["TAGS"])
    test_vectors = torch.Tensor(test_df["CONTENT"])
    test_labels = torch.Tensor(test_df["TAGS"])

    # Step 2: Create data loaders
    train_dataset = TensorDataset(train_vectors, train_labels)
    val_dataset = TensorDataset(val_vectors, val_labels)
    test_dataset = TensorDataset(test_vectors, test_labels)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # model = EmailRNN()

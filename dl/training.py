import os
from collections import Counter

import numpy as np
import pandas
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import TensorDataset, DataLoader

from dl.rnn import EmailRNN


class DatasetSplitHandler:
    def __init__(self, df: pandas.DataFrame):
        self.df = df
        self.mlb = MultiLabelBinarizer()
        self.tag_dis = df['TAGS'].value_counts()
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
        self.tag_dis.plot(
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
        }).reindex(self.tag_dis.index)

        split = split / split.sum(axis=0)
        split.plot(
            kind='bar',
            figsize=(10, 4),
            title='Tag Distribution per Split',
            ylabel='Proportion of observations'

        )

        plt.show()


def train(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs):
    model.train()
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model on the validation set
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()

        valid_loss /= len(valid_loader)
        # Check if the current validation loss is the best so far
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save the model or perform other actions as needed

        # Print the current epoch and validation loss
        print(f"Epoch {epoch + 1}/{num_epochs}: Validation Loss = {valid_loss:.4f}")

    # After training, evaluate the final model on the testing set
    accuracy, test_loss = evaluate(model, test_loader, criterion)
    print(f"Accuracy = {accuracy:.4f}")


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            outputs = torch.sigmoid(outputs)
            threshold = 0.6  # Adjust this threshold as needed
            predicted = (outputs >= threshold).int()

            # Convert tensors to numpy arrays
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(len(labels)):
                total += 1
                if np.array_equal(predicted[i], labels[i]):
                    correct += 1

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss


if __name__ == "__main__":
    df = pandas.read_json(os.path.join(os.getcwd(), "../data/preprocessed.json"), orient='index')
    dataset_split_handler = DatasetSplitHandler(df)
    dataset_split_handler.encode_labels()
    dataset_split_handler.split_dataset()

    # Convert preprocessed data into PyTorch tensors
    train_vectors = torch.Tensor(dataset_split_handler.email_train)
    train_tags = torch.Tensor(dataset_split_handler.tag_train.values)
    val_vectors = torch.Tensor(list(dataset_split_handler.email_val))
    val_tags = torch.Tensor(dataset_split_handler.tag_val.values)
    test_vectors = torch.Tensor(list(dataset_split_handler.email_test))
    test_tags = torch.Tensor(dataset_split_handler.tag_test.values)

    # Create data loaders
    train_dataset = TensorDataset(train_vectors, train_tags)
    val_dataset = TensorDataset(val_vectors, val_tags)
    test_dataset = TensorDataset(test_vectors, test_tags)

    train_loader = DataLoader(train_dataset, batch_size=100)
    val_loader = DataLoader(val_dataset, batch_size=100)
    test_loader = DataLoader(test_dataset, batch_size=100)

    # hyper-parameters
    num_epochs = 50
    learning_rate = 0.001

    input_size = 200
    hidden_size = 2
    num_layers = 1
    num_classes = 5

    rnn = EmailRNN(input_size, hidden_size, num_classes)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    train(rnn, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs)

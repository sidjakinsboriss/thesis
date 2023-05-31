import json
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score

from dl.dataset_split_handler import DatasetSplitHandler
from dl.rnn import EmailRNN


def train(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs):
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
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
    precision, recall, accuracy = evaluate(model, test_loader, criterion)
    print(f"Accuracy = {accuracy:.4f}")


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0

    predicted = []
    ground_truth = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            outputs = torch.sigmoid(outputs)
            threshold = 0.6  # Adjust this threshold as needed

            predicted.append((outputs >= threshold).int().cpu().numpy())
            ground_truth.append(labels.cpu().numpy())

            # Convert tensors to numpy arrays
            # labels += labels.cpu().numpy()

    predicted = np.concatenate(predicted, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    matrix(ground_truth, predicted)

    # Calculate precision, accuracy, and recall
    precision = precision_score(ground_truth, predicted, average='micro')
    accuracy = accuracy_score(ground_truth, predicted)
    recall = recall_score(ground_truth, predicted, average='micro')

    avg_loss = total_loss / len(test_loader)
    return precision, recall, accuracy


def matrix(ground_truth: np.ndarray, predicted: np.ndarray):
    row_names = col_names = np.array(['existence', 'not-ak', 'process', 'property', 'technology'])

    row_labels = np.unique(ground_truth, axis=0)
    column_labels = np.unique(predicted, axis=0)

    # Expand ground_truth labels
    for label in row_labels:
        indices = np.where(label == 1)[0]
        new_label = ', '.join([row_names[i] for i in indices])
        if new_label and new_label not in row_names:
            row_names = np.append(row_names, new_label)

    # Expand predicted labels
    for label in column_labels:
        indices = np.where(label == 1)[0]
        new_label = ', '.join([col_names[i] for i in indices])
        if new_label and new_label not in col_names:
            col_names = np.append(col_names, new_label)

    matrix = np.zeros((len(row_names), len(col_names)))

    for i, truth in enumerate(ground_truth):
        pred = predicted[i]

        indices = np.where(truth == 1)[0]
        row_label = ', '.join([row_names[i] for i in indices])
        row_index = np.where(row_names == row_label)[0][0]

        indices = np.where(pred == 1)[0]
        col_label = ', '.join([col_names[i] for i in indices])

        if col_label:
            col_index = np.where(col_names == col_label)[0][0]

            matrix[row_index][col_index] += 1

    num_rows, num_cols = matrix.shape
    fig, ax = plt.subplots()

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=90, ha='right')
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)

    # Create the heatmap
    heatmap = ax.matshow(matrix, cmap='Purples')
    fig.colorbar(heatmap)

    # Annotate the matrix values on the heatmap
    for i in range(num_rows):
        for j in range(num_cols):
            ax.text(j, i, str(matrix[i, j]), va='center', ha='center')

    # Save the plot
    plt.savefig("matrix.jpg", bbox_inches='tight')


if __name__ == "__main__":
    data = json.load(open(os.path.join(os.getcwd(), "../data/preprocessed.json"), "r"))
    df = pd.DataFrame.from_dict(data, orient="index")
    # df = pandas.read_json(os.path.join(os.getcwd(), "../data/preprocessed.json"), orient='index')
    dataset_split_handler = DatasetSplitHandler(df)
    dataset_split_handler.encode_labels()
    dataset_split_handler.split_dataset()

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

    for train_loader, val_loader, test_loader in dataset_split_handler.get_data_loaders():
        train(rnn, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs)

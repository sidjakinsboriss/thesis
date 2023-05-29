import os

import numpy as np
import pandas
import torch

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

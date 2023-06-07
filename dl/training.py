import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dl.dataset_split_handler import DatasetSplitHandler
from dl.rnn import EmailRNN
from dl.sampler import MultilabelBalancedRandomSampler

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


def draw_matrix(ground_truth: np.ndarray, predicted: np.ndarray):
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
        row_index = np.where(truth == 1)[0][0]
        col_index = np.where(predicted[i] == 1)[0][0]
        matrix[row_index][col_index] += 1

        indices = np.where(truth == 1)[0]
        row_name = ', '.join([row_names[i] for i in indices])
        row_index = np.where(row_names == row_name)[0][0]

        indices = np.where(predicted[i] == 1)[0]
        col_name = ', '.join([col_names[i] for i in indices])

        if col_name:
            col_index = np.where(col_names == col_name)[0][0]

            matrix[row_index][col_index] += 1

    num_rows, num_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=90, ha='right')
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)

    # Create the heatmap
    heatmap = ax.matshow(matrix, cmap='YlOrRd')
    fig.colorbar(heatmap)

    locator = MaxNLocator(nbins=len(col_names))
    ax.xaxis.set_major_locator(locator)
    locator = MaxNLocator(nbins=len(row_names))
    ax.yaxis.set_major_locator(locator)

    # Annotate the matrix values on the heatmap
    for i in range(num_rows):
        for j in range(num_cols):
            ax.text(j, i, str(matrix[i, j]), va='center', ha='center')

    # Save the plot
    plt.savefig("matrix.jpg", bbox_inches='tight')
    print("HI")


class Training:
    def __init__(self, model, criterion, optimizer, num_epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self, train_loader, test_loader, val_loader):
        self.model.to(device)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        best_valid_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step(loss)

            # Evaluate the model on the validation set
            self.model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    valid_loss += criterion(outputs, labels).item()

            valid_loss /= len(val_loader)
            # Check if the current validation loss is the best so far
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # Save the model or perform other actions as needed

            # Print the current epoch and validation loss
            print(f"Epoch {epoch + 1}/{num_epochs}: Validation Loss = {valid_loss:.4f}")

        # After training, evaluate the final model on the testing set
        self.evaluate(test_loader, criterion)

    def evaluate(self, test_loader, criterion):
        self.model.eval()
        total_loss = 0.0

        predicted = []
        ground_truth = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                outputs = torch.sigmoid(outputs)

                threshold = 0.5  # Adjust this threshold as needed

                outputs = torch.stack([(output > threshold).int() for output in outputs])

                predicted.append(outputs.cpu().numpy())
                ground_truth.append(labels.cpu().numpy())

        predicted = np.concatenate(predicted, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)

        draw_matrix(ground_truth, predicted)

        # Calculate precision, accuracy, and recall
        precision = precision_score(ground_truth, predicted, average='macro')
        accuracy = accuracy_score(ground_truth, predicted)
        recall = recall_score(ground_truth, predicted, average='macro')
        f1 = f1_score(ground_truth, predicted, average=None)
        f_score_average = f1_score(ground_truth, predicted, average='macro')

        print(f'F1 score per class: {f1}')
        print(f'F1 score average: {f_score_average}')


def plot_tag_distribution(df: pd.DataFrame):
    ax = df['TAGS'].value_counts().plot(
        kind='bar',
        figsize=(10, 4),
        title='Tag Distribution',
        ylabel='Proportion of observations'
    )
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()


def create_train_loader(train_loader: DataLoader) -> DataLoader:
    # tags = train_loader.dataset.labels
    # class_frequencies = tags.sum(dim=0)
    # weights = 1.0 / class_frequencies
    # sample_weights = np.dot(tags, weights)
    sampler = MultilabelBalancedRandomSampler(train_loader.dataset.labels)

    return DataLoader(dataset=train_loader.dataset,
                      batch_size=128,
                      sampler=sampler,
                      collate_fn=train_loader.collate_fn)


if __name__ == "__main__":
    # data = json.load(open(os.path.join(os.getcwd(), "../data/preprocessed.json"), "r"))
    # df = pd.DataFrame.from_dict(data, orient="index")
    # df.to_csv(os.path.join(os.getcwd(), "../data/dataframe.csv"))
    df = pd.read_csv(os.path.join(os.getcwd(), "../data/dataframe.csv"))

    dataset_split_handler = DatasetSplitHandler(df)
    dataset_split_handler.encode_labels()
    dataset_split_handler.split_dataset()
    # plot_tag_distribution(df)

    # hyper-parameters
    num_epochs = 1
    learning_rate = 0.0001

    input_size = 200
    hidden_size = 128
    num_layers = 1
    num_classes = 5
    dropout = 0.5

    weights = dataset_split_handler.weights

    rnn = EmailRNN(input_size, hidden_size, num_classes, num_layers, dropout, weights)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = BCEWithLogitsLoss()

    # train_loader = DataLoader(dataset=train_loader.dataset,
    #                           batch_size=8)
    #
    # training.train(train_loader, test_loader, val_loader)

    # Initialize loaders
    # train_loader = torch.load(os.path.join(os.getcwd(), "../data/training_loader.pt"))
    # train_loader = create_train_loader(train_loader)
    # val_loader = torch.load(os.path.join(os.getcwd(), "../data/val_loader.pt"))
    # test_loader = torch.load(os.path.join(os.getcwd(), "../data/test_loader.pt"))

    training = Training(rnn, criterion, optimizer, num_epochs)
    # training.train(train_loader, test_loader, val_loader)

    for train_loader, val_loader, test_loader in dataset_split_handler.get_data_loaders():
        torch.save(train_loader, os.path.join(os.getcwd(), "../data/training_loader.pt"))
        torch.save(val_loader, os.path.join(os.getcwd(), "../data/val_loader.pt"))
        torch.save(test_loader, os.path.join(os.getcwd(), "../data/test_loader.pt"))

        train_loader = create_train_loader(train_loader)
        training.train(train_loader, test_loader, val_loader)

import collections
import json
import os

import numpy as np
import pandas
import pandas as pd
import torch
from gensim.models import Word2Vec, KeyedVectors
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import f1_score, multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from dl.dataset_split_handler import DatasetHandler, TextDataset
from dl.loss import WBCEWithLogitLoss
from dl.rnn import EmailLSTM
from dl.sampler import MultilabelBalancedRandomSampler
from dl.utils import count_word_occurrences, get_pos_weight, plot_label_frequencies

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def draw_matrix(ground_truth: np.ndarray, predicted: np.ndarray):
    """
    Draws a symmetric matrix, where rows represent ground truth labels,
    and columns represent predicted labels
    """
    row_names = col_names = np.array(['existence', 'not-ak', 'process', 'property', 'technology'])

    row_labels = np.unique(ground_truth, axis=0)
    column_labels = np.unique(predicted, axis=0)

    for label in row_labels:
        indices = np.where(label == 1)[0]
        new_label = ', '.join([row_names[i] for i in indices])
        if new_label and new_label not in row_names:
            row_names = np.append(row_names, new_label)
            col_names = np.append(col_names, new_label)

    for label in column_labels:
        indices = np.where(label == 1)[0]
        new_label = ', '.join([col_names[i] for i in indices])
        if new_label and new_label not in col_names:
            row_names = np.append(row_names, new_label)
            col_names = np.append(col_names, new_label)

    matrix = np.zeros((len(row_names), len(col_names)))

    for i, truth in enumerate(ground_truth):
        pred = predicted[i]
        indices = np.where(pred == 1)[0]

        if indices.size != 0:
            col_name = ', '.join([col_names[i] for i in indices])

            indices = np.where(truth == 1)[0]
            row_name = ', '.join([row_names[i] for i in indices])
            row_index = np.where(row_names == row_name)[0][0]

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


def plot_email_word_counts(df: pandas.DataFrame):
    length_ranges = [x * 100 for x in range(60)]
    df['email_length'] = df['CONTENT'].apply(lambda x: len(x))
    df['length_range'] = pd.cut(df['email_length'], bins=length_ranges)
    email_count = df['length_range'].value_counts().sort_index()

    plt.bar(email_count.index.astype(str), email_count.values)
    plt.xlabel('Length Range')
    plt.ylabel('Number of Emails')
    plt.title('Email Count in Length Ranges')
    plt.xticks(rotation=45)
    plt.show()


def get_word_embeddings(path: str, use_trained_weights=True) -> KeyedVectors:
    """
    @param use_trained_weights: Whether to use weights obtained from the whole email dataset
    @return: Vectors containing words with their corresponding embeddings
    """
    if use_trained_weights:
        model = Word2Vec.load(path)
        keyed_vectors = model.wv
    else:
        keyed_vectors = KeyedVectors.load_word2vec_format(path, binary=True)

    padding_word = '<PAD>'
    padding_vector = torch.zeros(keyed_vectors.vector_size)

    updated_words = [padding_word] + keyed_vectors.index_to_key
    keyed_vectors_tensor = torch.tensor(keyed_vectors.vectors)

    updated_vectors = torch.cat((padding_vector.unsqueeze(0), keyed_vectors_tensor), dim=0)
    updated_key_to_index = {word: index for index, word in enumerate(updated_words)}

    keyed_vectors = KeyedVectors(vector_size=keyed_vectors.vector_size)
    keyed_vectors.vectors = updated_vectors.numpy()
    keyed_vectors.index_to_key = updated_words
    keyed_vectors.key_to_index = updated_key_to_index

    return keyed_vectors


class Training:
    def __init__(self, model, criterion, optimizer, num_epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self, train_loader, val_loader):
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        best_valid_loss = float('inf')
        best_f_score = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, labels, lengths in train_loader:
                inputs.to(device)
                labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs, lengths)

                loss = self.criterion(outputs, labels)

                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            scheduler.step(loss)

            train_loss /= len(train_loader)

            # Evaluate the model on the validation set
            self.model.eval()

            valid_loss = 0.0
            predicted = []
            ground_truth = []

            with torch.no_grad():
                for inputs, labels, lengths in val_loader:
                    inputs.to(device)
                    labels.to(device)

                    outputs = self.model(inputs, lengths)
                    loss = self.criterion(outputs, labels)

                    valid_loss += loss.item()

                    # Track f-score
                    outputs = torch.sigmoid(outputs)
                    threshold = 0.5  # Adjust this threshold as needed
                    outputs = torch.stack([(output > threshold).int() for output in outputs])
                    predicted.append(outputs.cpu().numpy())
                    ground_truth.append(labels.cpu().numpy())

            predicted = np.concatenate(predicted, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            # f_score_macro = f1_score(ground_truth, predicted, average='macro')

            valid_loss /= len(val_loader)
            # Check if the current validation loss is the best so far
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # Save the model or perform other actions as needed
                torch.save(self.model.state_dict(), os.path.join(os.getcwd(), 'models/lstm.pth'))

            # Print the current epoch and validation loss
            print(
                f'Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}')

    def evaluate(self, test_loader):
        self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models/lstm.pth')))
        self.model.eval()

        predicted = []
        ground_truth = []

        with torch.no_grad():
            for inputs, labels, lengths in test_loader:
                inputs.to(device)
                labels.to(device)

                outputs = self.model(inputs, lengths)
                outputs = torch.sigmoid(outputs)

                threshold = 0.5  # Adjust this threshold as needed

                outputs = torch.stack([(output > threshold).int() for output in outputs])

                predicted.append(outputs.cpu().numpy())
                ground_truth.append(labels.cpu().numpy())

        predicted = np.concatenate(predicted, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)

        print(classification_report(ground_truth, predicted))

        # Calculate F-score
        f1 = f1_score(ground_truth, predicted, average=None)
        f_score_micro = f1_score(ground_truth, predicted, average='micro')
        f_score_macro = f1_score(ground_truth, predicted, average='macro')
        f_score_weighted = f1_score(ground_truth, predicted, average='weighted')

        print(f'F1 score per class: {f1}')
        print(f'F1 score micro: {f_score_micro}')
        print(f'F1 score macro: {f_score_macro}')
        print(f'F1 score weighted: {f_score_weighted}')

        return ground_truth, predicted


if __name__ == "__main__":
    # data = json.load(open(os.path.join(os.getcwd(), "../data/unprocessed.json"), "r"))
    # df = pd.DataFrame.from_dict(data, orient="index")
    # df.to_csv(os.path.join(os.getcwd(), "../data/dataframe_unprocessed.csv"))
    df = pd.read_csv(os.path.join(os.getcwd(), '../data/dataframe.csv'))

    word_embeddings = get_word_embeddings(use_trained_weights=True)
    include_parent_email = False

    dataset_handler = DatasetHandler(df, word_embeddings.key_to_index, include_parent_email)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    # hyper-parameters
    num_epochs = 20
    learning_rate = 0.0002
    input_size = 200
    hidden_size = 128
    num_layers = 2
    num_classes = 5
    dropout = 0.5
    batch_size = 32

    # predicted = []
    # ground_truth = []

    for train_loader, val_loader, test_loader in dataset_handler.get_data_loaders(batch_size):
        # train_loader = create_train_loader(train_loader)
        # plot_label_frequencies(train_loader.dataset.labels)
        rnn = EmailLSTM(input_size, hidden_size, num_classes, num_layers, dropout,
                        torch.tensor(word_embeddings.vectors))
        rnn.to(device)

        optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate)
        criterion = BCEWithLogitsLoss(pos_weight=get_pos_weight(train_loader))

        training = Training(rnn, criterion, optimizer, num_epochs)

        training.train(train_loader, val_loader)
        truth, pred = training.evaluate(test_loader)

        draw_matrix(truth, pred)

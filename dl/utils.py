import collections

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report

from dl.constants import TAGS


def plot_model(model, name: str):
    """
    Plots the model architecture.
    @param model: a TensorFlow model
    @param name: name of the model (used to save an image)
    """
    tf.keras.utils.plot_model(model, to_file=f'{name}.png', show_shapes=True,
                              expand_nested=True,
                              show_layer_activations=True)


def plot_tag_distribution(df: pd.DataFrame):
    ax = df['TAGS'].value_counts().plot(
        kind='bar',
        figsize=(10, 4),
        ylabel='Number of observations'
    )
    for p in ax.patches:
        ax.annotate(str(p.get_height()),
                    (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.savefig('manual.jpg', bbox_inches='tight')


def plot_label_frequencies(labels, dataset: str = 'manual'):
    """
    Plots the amount of each individual label.
    """
    label_frequencies = np.sum(labels, axis=0)

    label_indices = np.arange(5)
    bars = plt.bar(label_indices, label_frequencies, edgecolor='black')

    plt.ylabel('Frequency')
    plt.xlabel('Label')

    plt.xticks(label_indices, TAGS)
    plt.grid(axis='y', alpha=0.75)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 3, yval + 3, int(yval),
                 va='bottom')

    plt.savefig(f'analysis/images/label_frequencies_{dataset}.jpg', bbox_inches='tight')
    plt.clf()


def plot_dataset_label_combination_frequencies(labels, dataset: str = 'manual'):
    """
    Plots the amount of label combinations.
    """
    # Count unique label combinations
    label_counts = collections.defaultdict(int)

    # Discard all single labels
    labels = [label for label in labels if len(label) > 1]

    for label in labels:
        label_counts[', '.join(label)] += 1

    # Plot the counts
    labels, counts = zip(*label_counts.items())

    zipped_lists = zip(counts, labels)
    zipped_lists_sorted = sorted(zipped_lists, reverse=True)
    counts, labels = zip(*zipped_lists_sorted)

    x = range(len(labels))
    bars = plt.bar(x, counts, tick_label=labels)
    plt.xticks(rotation=90)

    plt.ylabel('Frequency')
    plt.xlabel('Label combination')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 4.0, yval + 5, int(yval),
                 va='bottom', rotation=90)

    plt.ylim(0, max(counts) * 1.2)
    plt.savefig(f'analysis/images/label_combination_frequencies_{dataset}.jpg', bbox_inches='tight')
    plt.clf()


def plot_email_word_counts(df: pd.DataFrame):
    """
    Plots the amount of emails for different word count ranges.
    """
    length_ranges = [x * 100 for x in range(20)]
    df['email_length'] = df['BODY'].apply(lambda x: len(x))
    df['length_range'] = pd.cut(df['email_length'], bins=length_ranges)
    email_count = df['length_range'].value_counts().sort_index()

    bars = plt.bar(email_count.index.astype(str), email_count.values)
    plt.xticks(rotation=90)

    plt.ylabel('Number of emails')
    plt.xlabel('Email length range')

    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 4.0, y_val, int(y_val),
                 va='bottom', rotation=90)

    plt.ylim(0, max(email_count) * 1.2)
    plt.savefig('word_counts.jpg', bbox_inches='tight')
    plt.clf()


def get_embedding_matrix(embedding_dim, word_index, use_so=False):
    """
    Generates the embedding matrix that will be used to get the necessary word embeddings during the training.
    @param embedding_dim: length of word embeddings
    @param word_index: a dictionary where keys are the words and values are the indices into the embedding matrix
    @param use_so: whether to use the embeddings trained on Stack Overflow posts
    @return: embedding matrix
    """
    if use_so:
        wv = KeyedVectors.load_word2vec_format('dl/embeddings/SO_vectors_200.bin',
                                               binary=True)
    else:
        gensim_model = Word2Vec.load('dl/embeddings/word2vec_model')
        wv = gensim_model.wv

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in wv:
            embedding_matrix[i] = wv[word]
        else:
            # Create a random embedding for OOV words
            embedding_matrix[i] = np.random.uniform(-0.1, 0.1, embedding_dim)

    return embedding_matrix


def draw_matrix(ground_truth: np.ndarray, predicted: np.ndarray):
    """
    Draws a symmetric matrix, where rows represent ground truth labels,
    and columns represent predicted labels
    """
    row_names = col_names = np.array(TAGS)

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

    matrix = np.zeros((len(row_names), len(col_names)), dtype=np.int)

    for pred, truth in zip(predicted, ground_truth):
        indices = np.where(pred == 1)[0]
        col_name = ', '.join([col_names[i] for i in indices])

        indices = np.where(truth == 1)[0]
        row_name = ', '.join([row_names[i] for i in indices])

        row_index = np.where(row_names == row_name)[0][0]
        col_index = np.where(col_names == col_name)[0][0]

        matrix[row_index][col_index] += 1

    num_rows, num_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create the heatmap
    heatmap = ax.matshow(matrix, cmap='YlOrRd')
    fig.colorbar(heatmap)

    locator = MaxNLocator(nbins=len(row_names))
    ax.xaxis.set_major_locator(locator)

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=90, ha='right')
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)

    # Annotate the matrix values on the heatmap
    for i in range(num_rows):
        for j in range(num_cols):
            ax.text(j, i, str(matrix[i, j]), va='center', ha='center')

    # Save the plot
    plt.savefig('matrix.jpg', bbox_inches='tight')


def display_results(ground_truth, predicted):
    """
    Prints accuracy, recall, and f1-score metrics based on
    ground truth and predicted labels
    """
    print(classification_report(ground_truth, predicted, zero_division=1.0))


def transform_output(ground_truth: np.ndarray, predicted: np.ndarray):
    """
    Transforms the truth and predicted labels such that 'not-ak' is treated
    as a separate label when calculating performance metrics.
    """
    ground_truth_transformed = np.zeros((ground_truth.shape[0], 5), dtype=np.int8)
    predicted_transformed = np.zeros((predicted.shape[0], 5), dtype=np.int8)

    for i in range(len(ground_truth)):
        if np.array_equal(ground_truth[i], [0, 0, 0, 0]):
            ground_truth_transformed[i] = np.append(ground_truth[i], 1)
        else:
            ground_truth_transformed[i] = np.append(ground_truth[i], 0)

        if np.array_equal(predicted[i], [0, 0, 0, 0]):
            predicted_transformed[i] = np.append(predicted[i], 1)
        else:
            predicted_transformed[i] = np.append(predicted[i], 0)

    return ground_truth_transformed, predicted_transformed


def generate_class_weights(labels):
    """
    Generates class weights that will be applied to the loss function
    during the training process.
    @param labels: training labels
    @return: class weight for each of the 4 labels
    """
    class_counts = np.sum(labels, axis=0)

    class_weight = dict()
    for i, count in enumerate(class_counts):
        class_weight[i] = (1 / count) * (len(labels) / 2.0)
    return class_weight

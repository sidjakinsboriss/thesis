import collections
import json
from collections import Counter

import numpy as np
import pandas
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, \
    classification_report

from dl.constants import TAGS


def count_word_occurrences(df: pandas.DataFrame):
    text = ' '.join(df['CONTENT'].values)

    words = text.split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1],
                                reverse=True)

    word_counts_dict = {word: count for word, count in sorted_word_counts}

    # Export the word counts to JSON
    with open('word_counts.json', 'w') as f:
        json.dump(word_counts_dict, f, indent=4)


def draw_class_confusion_matrices(ground_truth, predicted):
    mcm = multilabel_confusion_matrix(ground_truth, predicted)

    # Loop through the confusion matrices and plot them
    for i, cm in enumerate(mcm):
        cmd = ConfusionMatrixDisplay(cm,
                                     display_labels=[f'Not {TAGS[i]}',
                                                     TAGS[i]])
        cmd.plot()
        cmd.ax_.set(title=f'Confusion Matrix for {TAGS[i]}',
                    xlabel='Predicted', ylabel='Actual')
        plt.savefig(f"confusion_matrix_{TAGS[i]}.jpg",
                    bbox_inches='tight')


def get_class_weights(df: pandas.DataFrame):
    labels = TAGS
    N = len(df)

    class_weights = {}
    positive_weights = {}
    negative_weights = {}

    for label in labels:
        positive_weights[label] = N / (2 * sum(df[label] == 1))
        negative_weights[label] = N / (2 * sum(df[label] == 0))

    class_weights['positive_weights'] = positive_weights
    class_weights['negative_weights'] = negative_weights

    return class_weights


def plot_tag_distribution(df: pd.DataFrame):
    ax = df['TAGS'].value_counts().plot(
        kind='bar',
        figsize=(10, 4),
        title='Tag Distribution',
        ylabel='Number of observations'
    )
    for p in ax.patches:
        ax.annotate(str(p.get_height()),
                    (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.savefig("manual.jpg", bbox_inches='tight')


def plot_label_frequencies(labels):
    label_frequencies = np.sum(labels, axis=0)

    label_indices = np.arange(5)
    bars = plt.bar(label_indices, label_frequencies, edgecolor='black')
    plt.ylabel('Frequency')
    plt.xticks(label_indices, TAGS)
    plt.grid(axis='y', alpha=0.75)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.5, yval, int(yval),
                 va='bottom')

    plt.show()


def plot_dataset_tag_combination_counts(labels):
    # Count unique label combinations
    label_counts = collections.defaultdict(int)
    for i in range(len(labels)):
        label = labels[i]
        email_tags = [TAGS[i] for i in range(5) if label[i] == 1]

        if len(email_tags) > 1:
            label = ', '.join(email_tags)
            label_counts[label] += 1

    # Plot the counts
    labels, counts = zip(*label_counts.items())

    zipped_lists = zip(counts, labels)
    zipped_lists_sorted = sorted(zipped_lists, reverse=True)
    counts, labels = zip(*zipped_lists_sorted)

    x = range(len(labels))
    bars = plt.bar(x, counts, tick_label=labels)
    plt.xticks(rotation=90)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 4.0, yval + 3, int(yval),
                 va='bottom', rotation=90)

    plt.ylim(0, max(counts) * 1.2)
    plt.savefig('tag_combination_counts.jpg', bbox_inches='tight')


def plot_email_word_counts(df: pd.DataFrame):
    length_ranges = [x * 100 for x in range(20)]
    df['email_length'] = df['CONTENT'].apply(lambda x: len(x))
    df['length_range'] = pd.cut(df['email_length'], bins=length_ranges)
    email_count = df['length_range'].value_counts().sort_index()

    bars = plt.bar(email_count.index.astype(str), email_count.values)
    plt.xticks(rotation=45)

    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 4.0, y_val + 10, int(y_val),
                 va='bottom', rotation=90)

    plt.ylim(0, max(email_count) * 1.2)
    plt.show()


def get_embedding_matrix(embedding_dim, word_index, use_so=True):
    if use_so:
        wv = KeyedVectors.load_word2vec_format('embeddings/SO_vectors_200.bin',
                                               binary=True)
    else:
        gensim_model = Word2Vec.load('embeddings/word2vec_model')
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

    row_names = np.append(row_names, 'not-ak')
    col_names = np.append(col_names, 'not-ak')

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
        if indices.size != 0:
            col_name = ', '.join([col_names[i] for i in indices])

            indices = np.where(truth == 1)[0]
            row_name = ', '.join([col_names[i] for i in indices])

            row_index = np.where(row_names == row_name)[0][0]
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
    plt.savefig('matrix.jpg', bbox_inches='tight')


def display_results(ground_truth, predicted):
    """
    Prints accuracy, recall, and f1-score metrics based on
    ground truth and predicted labels
    """
    print(classification_report(ground_truth, predicted))


def generate_class_weights(labels):
    class_counts = np.sum(labels, axis=0)

    class_weight = dict()
    for i, count in enumerate(class_counts):
        class_weight[i] = (1 / count) * (len(labels) / 2.0)
    return class_weight

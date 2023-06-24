import collections
import json
from collections import Counter

import numpy as np
import pandas
import pandas as pd
import torch
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score


def count_word_occurrences(df: pandas.DataFrame):
    text = ' '.join(df['CONTENT'].values)

    words = text.split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    word_counts_dict = {word: count for word, count in sorted_word_counts}

    # Export the word counts to JSON
    with open('word_counts.json', 'w') as f:
        json.dump(word_counts_dict, f, indent=4)


def draw_class_confusion_matrices(predicted, ground_truth):
    mcm = multilabel_confusion_matrix(predicted, ground_truth)

    # Define the display labels for your problem
    display_labels = ['existence', 'not-ak', 'process', 'property', 'technology']

    # Loop through the confusion matrices and plot them
    for i, cm in enumerate(mcm):
        cmd = ConfusionMatrixDisplay(cm, display_labels=[f'Not {display_labels[i]}', display_labels[i]])
        cmd.plot()
        cmd.ax_.set(title=f'Confusion Matrix for {display_labels[i]}', xlabel='Predicted', ylabel='Actual')
        plt.show()


def get_class_weights(df: pandas.DataFrame):
    labels = ['existence', 'not-ak', 'process', 'property', 'technology']
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


def get_pos_weight(train_loader):
    """
    Calculates weights of positive samples for each class
    """
    labels = train_loader.dataset.labels

    num_positives = torch.sum(labels, dim=0)
    num_negatives = len(labels) - num_positives

    return num_negatives / num_positives


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


def plot_label_frequencies(labels):
    label_frequencies = torch.sum(labels, dim=0)

    label_indices = np.arange(5)
    plt.bar(label_indices, label_frequencies.numpy(), edgecolor='black')
    plt.xlabel('Label Index')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels')
    plt.xticks(label_indices)
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def plot_dataset_tag_combination_counts(dataset):
    tags = ['existence', 'not-ak', 'process', 'property', 'technology']

    # Count unique label combinations
    label_counts = collections.defaultdict(int)
    for i in range(len(dataset)):
        label = dataset.labels[i]
        label = ', '.join([tags[i] for i in range(5) if label[i] == 1])
        label_counts[label] += 1

    # Plot the counts
    labels, counts = zip(*label_counts.items())
    x = range(len(labels))
    plt.bar(x, counts, tick_label=labels)
    plt.xticks(rotation=90)
    plt.xlabel("Label combinations")
    plt.ylabel("Counts")

    plt.savefig("test.jpg", bbox_inches='tight')


def get_embedding_matrix(embedding_dim, word_index):
    gensim_model = Word2Vec.load('embeddings/word2vec_model')
    wv = gensim_model.wv

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in wv:
            embedding_matrix[i] = wv[word]

    return embedding_matrix


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


def display_results(ground_truth, predicted):
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


import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


def generate_class_weights(class_series, multi_class=True, one_hot_encoded=True):
    """
    Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
    Some examples of different formats of class_series and their outputs are:
      - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
      {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
      - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
      {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
      - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
      {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
      - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
      {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
    The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
    of appareance of the label when the dataset was processed.
    In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
    Author: Angel Igareta (angel@igareta.com)
    """
    if multi_class:
        # If class is one hot encoded, transform to categorical labels to use compute_class_weight
        if one_hot_encoded:
            class_series = np.argmax(class_series, axis=1)

        # Compute class weights with sklearn method
        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
        return dict(zip(class_labels, class_weights))
    else:
        # It is neccessary that the multi-label values are one-hot encoded
        mlb = None
        if not one_hot_encoded:
            mlb = MultiLabelBinarizer()
            class_series = mlb.fit_transform(class_series)

        n_samples = len(class_series)
        n_classes = len(class_series[0])

        # Count each class frequency
        class_count = [0] * n_classes
        for classes in class_series:
            for index in range(n_classes):
                if classes[index] != 0:
                    class_count[index] += 1

        # Compute class weights using balanced method
        class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
        class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
        return dict(zip(class_labels, class_weights))

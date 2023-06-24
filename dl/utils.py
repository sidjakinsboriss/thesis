import collections
import json
from collections import Counter

import numpy as np
import pandas
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from dl.dataset_split_handler import TextDataset
from dl.sampler import MultilabelBalancedRandomSampler


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


def create_train_loader(train_loader) -> DataLoader:
    sampler = MultilabelBalancedRandomSampler(train_loader.dataset.labels)

    return DataLoader(dataset=train_loader.dataset,
                      batch_size=64,
                      sampler=sampler,
                      collate_fn=train_loader.collate_fn,
                      drop_last=True)


def get_pos_weight(train_loader: DataLoader) -> torch.Tensor:
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


def plot_dataset_tag_combination_counts(dataset: TextDataset):
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

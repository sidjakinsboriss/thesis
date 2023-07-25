import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from dl.constants import TAGS


def draw_tag_counts(labels):
    label_frequencies = np.sum(labels, axis=0)

    label_indices = np.arange(5)
    bars = plt.bar(label_indices, label_frequencies, edgecolor='black')
    plt.ylabel('Frequency')
    plt.xticks(label_indices, TAGS)
    plt.grid(axis='y', alpha=0.75)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 3, yval, int(yval), va='bottom')

    plt.savefig("tag_counts.jpg", bbox_inches='tight')
    plt.clf()


def draw_label_percentages(labels):
    tags = [TAGS[i] for i in range(5) if i != 1]
    label_frequencies = np.sum(labels, axis=0)

    fig, ax = plt.subplots()
    ax.pie(label_frequencies, labels=tags, autopct='%1.1f%%')

    plt.savefig('pie_chart.jpg', bbox_inches='tight')
    plt.clf()


def draw_thread_sizes_per_label(df: pd.DataFrame):
    tags = [TAGS[i] for i in range(5) if i != 1]
    thread_names = df[df['subject'].map(lambda subject: 'Re:' not in subject)]['subject'].values
    thread_sizes = {tag: [] for tag in tags}

    for thread_name in thread_names:
        df_subset = df[df['subject'].str.contains(thread_name, regex=False)]
        thread_size = len(df_subset)

        thread_tags = df_subset['tags'].values.tolist()
        thread_tags = [item for sublist in thread_tags for item in sublist]

        for tag in tags:
            if tag in thread_tags:
                thread_sizes[tag].append(thread_size)

    plt.boxplot(thread_sizes.values(), 0, '', vert=True,
                patch_artist=True, labels=tags)

    plt.savefig('thread_sizes.jpg', bbox_inches='tight')
    plt.clf()


def draw_email_thread_participation(df: pd.DataFrame):
    tags = [TAGS[i] for i in range(5) if i != 1]
    thread_names = df[df['subject'].map(lambda subject: 'Re:' not in subject)]['subject'].values
    thread_sizes = {tag: [] for tag in tags}

    for thread_name in thread_names:
        df_subset = df[df['subject'].str.contains(thread_name, regex=False)]

        thread_participants = list(set(df_subset['sent_from'].values.tolist()))
        thread_participants = len(thread_participants)

        thread_tags = df_subset['tags'].values.tolist()
        thread_tags = [item for sublist in thread_tags for item in sublist]

        for tag in tags:
            if tag in thread_tags:
                thread_sizes[tag].append(thread_participants)

    plt.boxplot(thread_sizes.values(), 0, '', vert=True,
                patch_artist=True, labels=tags)

    plt.savefig('thread_participants.jpg', bbox_inches='tight')
    plt.clf()


def analyze(df: pd.DataFrame):
    mlb = MultiLabelBinarizer()

    df['tags'] = df['tags'].transform(
        lambda tags: [tag for tag in tags if tag in TAGS]
    )

    df = df[df['tags'].map(lambda d: len(d) > 0)]

    encoded = mlb.fit_transform(df['tags'])
    df[mlb.classes_] = encoded

    # draw_tag_counts(df[mlb.classes_])
    # plot_dataset_tag_combination_counts(df[mlb.classes_].values.tolist())
    # draw_label_percentages(df[np.delete(mlb.classes_, 1)])
    # draw_thread_sizes_per_label(df)
    draw_email_thread_participation(df)


if __name__ == '__main__':
    df_manual = pd.read_json('emails_manual.json')
    df_bert = pd.read_json('emails_bert.json')

    analyze(df_bert)

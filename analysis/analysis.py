import os

import jaydebeapi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from dl.constants import TAGS
from dl.unlabeled_email_classification import JDBC_DRIVER_PATH
from dl.utils import plot_label_frequencies, plot_dataset_label_combination_frequencies


def generate_bert_dataset():
    # Connect to the database
    db_path = os.path.join(os.getcwd(), 'data/dataset-bert/database')
    conn = jaydebeapi.connect('org.h2.Driver',
                              f'jdbc:h2:file:{db_path};IFEXISTS=TRUE;AUTO_SERVER=TRUE',
                              ['', ''],
                              JDBC_DRIVER_PATH)
    cursor = conn.cursor()

    # Create a dictionary with tag ids
    cursor.execute('SELECT * FROM TAG')
    tag_ids = [row[:2] for row in cursor.fetchall()]
    tag_ids_dict = {}

    for tag_id in tag_ids:
        tag_ids_dict[tag_id[0]] = tag_id[1]

    # Get all email information
    cursor.execute('SELECT ID, MESSAGE_ID, PARENT_ID, SUBJECT, SENT_FROM, BODY FROM EMAIL')
    emails = cursor.fetchall()
    emails = [(email[0], email[1], email[2], email[3], email[4], email[5], []) for email in emails]

    for email in emails:
        email_id = email[0]

        cursor.execute(f'SELECT TAG_ID FROM EMAIL_TAG WHERE EMAIL_ID = {email_id}')
        email_tag_ids = cursor.fetchall()

        email_tags = [tag_ids_dict[tag_id[0]] for tag_id in email_tag_ids]

        for tag in email_tags:
            email[6].append(tag)

    df = pd.DataFrame(emails, columns=['ID', 'MESSAGE_ID', 'PARENT_ID', 'SUBJECT', 'SENT_FROM', 'BODY', 'TAGS'])

    df['TAGS'] = df['TAGS'].transform(
        lambda x: [label for label in x if label in TAGS]
    )
    df['TAGS'] = df['TAGS'].transform(
        lambda x: ', '.join(x)
    )

    df.to_json('data/bert_classified_emails.json', orient='index', indent=4)


def draw_label_percentages(labels, dataset: str = 'manual'):
    """
    Plots the percentage of each label in a dataset.
    """
    tags = [TAGS[i] for i in range(5) if i != 4]
    label_frequencies = np.sum(labels, axis=0)

    fig, ax = plt.subplots()
    ax.pie(label_frequencies, labels=tags, autopct='%1.1f%%')

    plt.savefig(f'analysis/images/pie_chart_{dataset}.jpg', bbox_inches='tight')
    plt.clf()


def draw_thread_sizes_per_label(df: pd.DataFrame, dataset: str = 'manual'):
    """
    Plots a boxplot of the thread sizes for each label.
    """
    tags = [TAGS[i] for i in range(5) if i != 4]
    thread_names = df[df['SUBJECT'].map(lambda subject: 'Re:' not in subject)]['SUBJECT'].values
    thread_sizes = {tag: [] for tag in tags}

    for thread_name in thread_names:
        df_subset = df[df['SUBJECT'].str.contains(thread_name, regex=False)]
        thread_size = len(df_subset)

        thread_tags = df_subset['TAGS'].values.tolist()
        thread_tags = [item for sublist in thread_tags for item in sublist.split(',')]

        for tag in tags:
            if tag in thread_tags:
                thread_sizes[tag].append(thread_size)

    plt.ylabel('Thread size')
    plt.xlabel('Label')

    plt.boxplot(thread_sizes.values(), 0, '', vert=True,
                patch_artist=True, labels=tags)

    plt.savefig(f'analysis/images/thread_sizes_{dataset}.jpg', bbox_inches='tight')
    plt.clf()


def draw_email_thread_participation(df: pd.DataFrame, dataset: str = 'manual'):
    """
    Plots a boxplot of the number of thread participants for each label.
    """
    tags = [TAGS[i] for i in range(5) if i != 4]
    thread_names = df[df['SUBJECT'].map(lambda subject: 'Re:' not in subject)]['SUBJECT'].values
    thread_sizes = {tag: [] for tag in tags}

    for thread_name in thread_names:
        df_subset = df[df['SUBJECT'].str.contains(thread_name, regex=False)]

        thread_participants = list(set(df_subset['SENT_FROM'].values.tolist()))
        thread_participants = len(thread_participants)

        thread_tags = df_subset['TAGS'].values.tolist()
        thread_tags = [item for sublist in thread_tags for item in sublist.split(',')]

        for tag in tags:
            if tag in thread_tags:
                thread_sizes[tag].append(thread_participants)

    plt.ylabel('Number of participants')
    plt.xlabel('Label')

    plt.boxplot(thread_sizes.values(), 0, '', vert=True,
                patch_artist=True, labels=tags)

    plt.savefig(f'analysis/images/thread_participants_{dataset}.jpg', bbox_inches='tight')
    plt.clf()


def analyze(df: pd.DataFrame, dataset: str = 'manual'):
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df['TAGS'].str.split(', '))

    if dataset == 'bert':
        encoded = np.array([enc[1:] for enc in encoded])
        mlb.classes_ = mlb.classes_[1:]

    df[mlb.classes_] = encoded

    # Plots
    plot_label_frequencies(df[mlb.classes_], dataset)
    plot_dataset_label_combination_frequencies(df['TAGS'].str.split(', ').array, dataset)
    draw_label_percentages(df[np.delete(mlb.classes_, 1)], dataset)
    draw_thread_sizes_per_label(df, dataset)
    draw_email_thread_participation(df, dataset)


if __name__ == '__main__':
    df_manual = pd.read_json('data/labeled_dataset_preprocessed.json', orient='index')

    if not os.path.exists('data/bert_classified_emails.json'):
        generate_bert_dataset()

    df_bert = pd.read_json('data/bert_classified_emails.json', orient='index')

    analyze(df_manual)
    analyze(df_bert, 'bert')

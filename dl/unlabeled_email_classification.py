import os

import jaydebeapi
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertModel

from dl.classifiers.bert.training_bert import batch_encode
from dl.constants import MODEL_NAME, TAGS

JDBC_DRIVER_PATH = 'C:/Program Files (x86)/H2/bin/h2-2.1.214.jar'

if __name__ == '__main__':
    # Connect to the database
    db_path = os.path.join(os.getcwd(), 'data/dataset-bert/database')
    conn = jaydebeapi.connect('org.h2.Driver',
                              f'jdbc:h2:file:{db_path};IFEXISTS=TRUE;AUTO_SERVER=TRUE',
                              ['', ''],
                              JDBC_DRIVER_PATH)
    cursor = conn.cursor()

    # Get IDs of all tagged emails
    cursor.execute("SELECT EMAIL_ID FROM EMAIL_TAG")
    rows = cursor.fetchall()
    tagged_email_ids = sorted(list(set([row[0] for row in rows])))

    # Create a dictionary with tag ids
    cursor.execute("SELECT * FROM TAG")
    tag_ids = [row[:2] for row in cursor.fetchall()]

    tag_ids_dict = {}
    tag_ids_dict_reverse = {}

    for tag_id in tag_ids:
        tag_ids_dict[tag_id[1]] = tag_id[0]
        tag_ids_dict_reverse[tag_id[0]] = tag_id[1]

    cursor.execute("SELECT * FROM EMAIL")
    emails = [row for row in cursor.fetchall()]
    emails = [(email[0], email[1], email[3], email[5], []) for email in emails]

    for email in emails:
        email_id = email[0]

        cursor.execute(f"SELECT TAG_ID FROM EMAIL_TAG WHERE EMAIL_ID = {email_id}")
        email_tag_ids = cursor.fetchall()

        email_tags = [tag_ids_dict_reverse[tag_id[0]] for tag_id in email_tag_ids]

        for tag in email_tags:
            email[4].append(tag)

    # Load the pre-processed dataset
    df = pd.read_json('data/whole_dataset_preprocessed.json', orient='index')

    # Choose emails that are not tagged
    df = df[~df['ID'].isin(tagged_email_ids)]
    email_texts = df['BODY'].tolist()
    email_ids = df['ID'].tolist()

    # Replace nan values with an empty string
    for i, text in enumerate(email_texts):
        if not isinstance(text, str):
            email_texts[i] = ''

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    ids, attention = batch_encode(tokenizer, email_texts)

    # Load the BERT model
    model = tf.keras.models.load_model('bert.h5', custom_objects={"TFDistilBertModel": TFDistilBertModel})

    # Predict the labels
    output = model.predict([ids, attention])
    output = tf.math.sigmoid(output).numpy()

    threshold = 0.5
    predictions = (output >= threshold).astype(int)

    predicted_labels = []

    for label in predictions:
        indices = np.where(label == 1)[0]
        if indices.size != 0:
            predicted_label = ['bert'] + [TAGS[i] for i in indices]
        else:
            predicted_label = ['bert', 'not-ak']
        predicted_labels.append(predicted_label)

    # Insert new labeled emails into EMAIL_TAG table
    for i, labels in enumerate(predicted_labels):
        email_id = email_ids[i]
        for label in labels:
            tag_id = tag_ids_dict[label]
            query = f"INSERT INTO EMAIL_TAG (EMAIL_ID, TAG_ID) VALUES ({email_id}, {tag_id})"
            cursor.execute(query)

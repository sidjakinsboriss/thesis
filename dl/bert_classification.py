import os

import jaydebeapi
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertModel

from dl.training_bert import batch_encode
from dl.utils import TAGS

if __name__ == '__main__':
    # Connect to the database
    db_path = os.path.join(os.getcwd(), "../data/dataset-bert/database")
    conn = jaydebeapi.connect("org.h2.Driver",
                              f"jdbc:h2:file:{db_path};IFEXISTS=TRUE;AUTO_SERVER=TRUE",
                              ["", ""],
                              "C:/Program Files (x86)/H2/bin/h2-2.1.214.jar")
    cursor = conn.cursor()

    # Get IDs of all tagged emails
    cursor.execute("SELECT EMAIL_ID FROM EMAIL_TAG")
    rows = cursor.fetchall()
    tagged_email_ids = sorted(list(set([row[0] for row in rows])))

    # Create a dictionary with tag ids
    cursor.execute("SELECT * FROM TAG")
    tag_ids = [row[:2] for row in cursor.fetchall()]

    tag_ids_dict = {}

    for tag_id in tag_ids:
        tag_ids_dict[tag_id[1]] = tag_id[0]

    # Load the dataset
    df = pd.read_csv("../data/whole_dataset.csv")

    # Choose emails that are not tagged
    df = df[~df['ID'].isin(tagged_email_ids)]
    email_texts = df['CONTENT'].tolist()
    email_ids = df['ID'].tolist()

    # Replace nan values with an empty string
    for i, text in enumerate(email_texts):
        if not isinstance(text, str):
            email_texts[i] = ''

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # ids, attention = batch_encode(tokenizer, email_texts, 64)
    #
    # ids_numpy = ids.numpy()
    # attention_numpy = attention.numpy()
    #
    # # Save using numpy save
    # np.save('ids.npy', ids_numpy)
    # np.save('attention.npy', attention_numpy)

    # ids = np.load('ids.npy')
    # attention = np.load('attention.npy')
    #
    # ids = tf.convert_to_tensor(ids)
    # attention = tf.convert_to_tensor(attention)

    # Load the model
    # model = tf.keras.models.load_model('bert', custom_objects={"TFDistilBertModel": TFDistilBertModel})

    # Predict the labels
    # output = model.predict([ids, attention])
    # output = tf.math.sigmoid(output).numpy()
    #
    # threshold = 0.5
    # predictions = (output >= threshold).astype(int)

    predictions = np.load('predictions.npy')

    tags = TAGS

    predicted_labels = []

    for label in predictions:
        indices = np.where(label == 1)[0]
        if indices.size != 0:
            predicted_label = ['bert'] + [tags[i] for i in indices]
            predicted_labels.append(predicted_label)

    # Insert new labeled emails into EMAIL_TAG table
    for i, labels in enumerate(predicted_labels):
        email_id = email_ids[i]
        for label in labels:
            tag_id = tag_ids_dict[label]
            query = f"INSERT INTO EMAIL_TAG (EMAIL_ID, TAG_ID) VALUES ({email_id}, {tag_id})"
            cursor.execute(query)

import os

import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast

from classifiers.bert import BertMultiLabel
from dataset_split_handler import DatasetHandler

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 200

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.getcwd(), '../data/dataframe.csv'))
    dataset_handler = DatasetHandler(df, None, include_parent_email=False)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    train_indices, val_indices, test_indices = dataset_handler.get_indices()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_texts = df['CONTENT'].iloc[train_indices].tolist()
    val_texts = df['CONTENT'].iloc[val_indices].tolist()
    test_texts = df['CONTENT'].iloc[test_indices].tolist()

    train_tags = df[dataset_handler.mlb.classes_].iloc[train_indices].values
    val_tags = df[dataset_handler.mlb.classes_].iloc[val_indices].values
    test_tags = df[dataset_handler.mlb.classes_].iloc[test_indices].values

    train_encodings = tokenizer(train_texts, truncation=True, padding=True,
                                max_length=MAX_LENGTH, return_tensors="tf")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True,
                              max_length=MAX_LENGTH, return_tensors="tf")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True,
                               max_length=MAX_LENGTH, return_tensors="tf")

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_tags))
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_tags))
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_tags))

    model = BertMultiLabel().get_model()

    # Hyper-parameters
    batch_size = 32
    num_epochs = 1

    training_history = model.fit(
        train_dataset.batch(batch_size), epochs=num_epochs, batch_size=batch_size, validation_data=val_dataset
    )

    test_loss, test_accuracy = model.evaluate(
        test_dataset.batch(batch_size)
    )

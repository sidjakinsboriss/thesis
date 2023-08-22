import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast

from dl.classifiers.bert.bert import EmailBERTHyperModel
from dl.constants import MAX_SEQUENCE_LENGTH, MODEL_NAME
from dl.dataset_handler import DatasetHandler
from dl.utils import generate_class_weights


def batch_encode(tokenizer, texts, max_length=MAX_SEQUENCE_LENGTH):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks.
    """""""""
    inputs = tokenizer.batch_encode_plus(texts,
                                         max_length=max_length,
                                         padding='max_length',
                                         # implements dynamic padding
                                         truncation=True,
                                         return_attention_mask=True,
                                         return_token_type_ids=False
                                         )

    return tf.convert_to_tensor(inputs['input_ids']), tf.convert_to_tensor(inputs['attention_mask'])


if __name__ == '__main__':
    df = pd.read_json('data/labeled_dataset_preprocessed.json', orient='index')
    dataset_handler = DatasetHandler(df)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    # Hyper-parameters
    batch_size = 32
    max_epochs = 30

    train_indices, val_indices, test_indices = dataset_handler.get_indices_for_optimization(under_sample=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_texts = df['BODY'].iloc[train_indices].tolist()
    val_texts = df['BODY'].iloc[val_indices].tolist()
    test_texts = df['BODY'].iloc[test_indices].tolist()

    labels = np.delete(dataset_handler.mlb.classes_, 1)  # remove the not-ak label
    train_tags = df[labels].iloc[train_indices].values.astype('float32')
    val_tags = df[labels].iloc[val_indices].values.astype('float32')
    test_tags = df[labels].iloc[test_indices].values.astype('float32')

    train_ids, train_attention = batch_encode(tokenizer, train_texts)
    valid_ids, valid_attention = batch_encode(tokenizer, val_texts)
    test_ids, test_attention = batch_encode(tokenizer, test_texts)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner = kt.Hyperband(hypermodel=EmailBERTHyperModel(),
                         objective='val_loss',
                         max_epochs=max_epochs,
                         factor=3,
                         directory='bert_optimization',
                         project_name='bert_optimization')
    tuner.search([train_ids, train_attention], train_tags,
                 validation_data=([valid_ids, valid_attention], val_tags),
                 class_weight=generate_class_weights(train_tags),
                 callbacks=[stop_early])

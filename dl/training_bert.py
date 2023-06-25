import os

import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast

from classifiers.bert import BertMultiLabel
from dataset_split_handler import DatasetHandler

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 200


def batch_encode(tokenizer, texts, batch_size, max_length=MAX_LENGTH):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.

    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length',  # implements dynamic padding
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.getcwd(), '../data/dataframe.csv'))
    dataset_handler = DatasetHandler(df, None, include_parent_email=False)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    # Hyper-parameters
    batch_size = 32
    num_epochs = 1

    train_indices, val_indices, test_indices = dataset_handler.get_indices()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_texts = df['CONTENT'].iloc[train_indices].tolist()
    val_texts = df['CONTENT'].iloc[val_indices].tolist()
    test_texts = df['CONTENT'].iloc[test_indices].tolist()

    train_tags = df[dataset_handler.mlb.classes_].iloc[train_indices].values
    val_tags = df[dataset_handler.mlb.classes_].iloc[val_indices].values
    test_tags = df[dataset_handler.mlb.classes_].iloc[test_indices].values

    train_ids, train_attention = batch_encode(tokenizer, train_texts, batch_size)
    valid_ids, valid_attention = batch_encode(tokenizer, val_texts, batch_size)
    test_ids, test_attention = batch_encode(tokenizer, test_texts, batch_size)

    model = BertMultiLabel().get_model()

    training_history = model.fit(
        [train_ids, train_attention], train_tags, epochs=num_epochs, batch_size=batch_size,
        validation_data=([valid_ids, valid_attention], val_tags)
    )

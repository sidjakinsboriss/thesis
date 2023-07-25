import os
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast

from dl.classifiers.bert.bert import EmailBERT
from dl.constants import MAX_SEQUENCE_LENGTH, MODEL_NAME
from dl.dataset_handler import DatasetHandler
from dl.loss import calculate_class_weights
from dl.utils import display_results, draw_matrix


def batch_encode(tokenizer, texts, batch_size, max_length=MAX_SEQUENCE_LENGTH):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks.
    """""""""

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length',
                                             # implements dynamic padding
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


if __name__ == '__main__':
    # df = pd.read_json(os.path.join(os.getcwd(), '../data/unprocessed.json'), orient='index')
    # df.to_csv(os.path.join(os.getcwd(), '../data/dataframe_unprocessed.csv'), index=None)

    df = pd.read_csv(
        os.path.join(os.getcwd(), '../data/dataframe_unprocessed.csv'))
    dataset_handler = DatasetHandler(df)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    calculate_class_weights(df)

    # Hyper-parameters
    batch_size = 32
    num_epochs = 1
    lr = 0.0001

    tags = []
    pred = []

    for train_indices, val_indices, test_indices in dataset_handler.get_indices():
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

        train_texts = df['CONTENT'].iloc[train_indices].tolist()
        val_texts = df['CONTENT'].iloc[val_indices].tolist()
        test_texts = df['CONTENT'].iloc[test_indices].tolist()

        train_indices_parent = dataset_handler.get_parent_indices(train_indices)
        val_indices_parent = dataset_handler.get_parent_indices(val_indices)
        test_indices_parent = dataset_handler.get_parent_indices(test_indices)

        train_parent_texts = df['CONTENT'].iloc[train_indices_parent].tolist()
        val_parent_texts = df['CONTENT'].iloc[val_indices_parent].tolist()
        test_parent_texts = df['CONTENT'].iloc[test_indices_parent].tolist()

        train_texts = [
            (train_parent_texts[i], train_texts[i]) if train_parent_texts[i] != train_texts[i] else train_texts[i] for i
            in range(len(train_texts))]
        val_texts = [
            (val_parent_texts[i], val_texts[i]) if val_parent_texts[i] != val_texts[i] else val_texts[i] for i in
            range(len(val_texts))]
        test_texts = [(test_parent_texts[i], test_texts[i]) if test_parent_texts[i] != test_texts[i] else test_texts[i]
                      for i in range(len(test_texts))]

        train_tags = df[dataset_handler.mlb.classes_].iloc[
            train_indices].values.astype('float32')
        val_tags = df[dataset_handler.mlb.classes_].iloc[
            val_indices].values.astype('float32')
        test_tags = df[dataset_handler.mlb.classes_].iloc[
            test_indices].values.astype('float32')

        train_ids, train_attention = batch_encode(tokenizer, train_texts,
                                                  batch_size)
        valid_ids, valid_attention = batch_encode(tokenizer, val_texts,
                                                  batch_size)
        test_ids, test_attention = batch_encode(tokenizer, test_texts,
                                                batch_size)

        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
        model = EmailBERT().get_model()

        # tf.keras.utils.plot_model(model, to_file='bert.png', show_shapes=True,
        #                           expand_nested=True,
        #                           show_layer_activations=True)

        model.fit(
            [train_ids, train_attention], train_tags, epochs=num_epochs,
            batch_size=batch_size,
            validation_data=([valid_ids, valid_attention], val_tags)
        )

        # model = tf.keras.models.load_model('bert', custom_objects={"TFDistilBertModel": TFDistilBertModel})
        test_output = model.predict([test_ids, test_attention])
        # test_output = tf.math.sigmoid(test_output).numpy()

        threshold = 0.5
        predicted_labels = (test_output >= threshold).astype(int)

        tags.append(test_tags)
        pred.append(predicted_labels)

        tags = np.array(list(chain.from_iterable(tags)))
        pred = np.array(list(chain.from_iterable(pred)))

        display_results(tags, pred)
        draw_matrix(tags, pred)

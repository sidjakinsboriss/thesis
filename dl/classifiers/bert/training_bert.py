import os
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from dl.classifiers.bert.bert import EmailBERT
from dl.constants import MAX_SEQUENCE_LENGTH, MODEL_NAME, TAGS
from dl.dataset_handler import DatasetHandler
from dl.utils import display_results, draw_matrix, generate_class_weights
from sklearn.utils import compute_sample_weight
from transformers import DistilBertTokenizerFast


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
    df = pd.read_csv(os.path.join(os.getcwd(), 'data/dataframe.csv'))
    dataset_handler = DatasetHandler(df)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    # Hyper-parameters
    batch_size = 64
    num_epochs = 1
    lr = 0.0001

    tags = []
    pred = []

    include_parent = False

    for train_indices, val_indices, test_indices in dataset_handler.get_indices(under_sample=True):
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

        train_texts = df['CONTENT'].iloc[train_indices].tolist()
        val_texts = df['CONTENT'].iloc[val_indices].tolist()
        test_texts = df['CONTENT'].iloc[test_indices].tolist()

        if include_parent:
            train_indices_parent = dataset_handler.get_parent_indices(train_indices)
            val_indices_parent = dataset_handler.get_parent_indices(val_indices)
            test_indices_parent = dataset_handler.get_parent_indices(test_indices)

            train_parent_texts = df['CONTENT'].iloc[train_indices_parent].tolist()
            val_parent_texts = df['CONTENT'].iloc[val_indices_parent].tolist()
            test_parent_texts = df['CONTENT'].iloc[test_indices_parent].tolist()

            train_texts = [
                (train_parent_texts[i], train_texts[i]) if train_parent_texts[i] != train_texts[i] else train_texts[i]
                for i
                in range(len(train_texts))]
            val_texts = [
                (val_parent_texts[i], val_texts[i]) if val_parent_texts[i] != val_texts[i] else val_texts[i] for i in
                range(len(val_texts))]
            test_texts = [
                (test_parent_texts[i], test_texts[i]) if test_parent_texts[i] != test_texts[i] else test_texts[i]
                for i in range(len(test_texts))]

        train_ids, train_attention = batch_encode(tokenizer, train_texts)
        valid_ids, valid_attention = batch_encode(tokenizer, val_texts)
        test_ids, test_attention = batch_encode(tokenizer, test_texts)

        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        positive_weights = {}
        negative_weights = {}

        N = len(df)

        for label in sorted(TAGS):
            positive_weights[label] = N / (2 * sum(df[label] == 1))
            negative_weights[label] = N / (2 * sum(df[label] == 0))

        labels = dataset_handler.mlb.classes_
        train_tags = df[labels].iloc[
            train_indices].values.astype('float32')

        class_weight = [{0: neg_weight, 1: pos_weight} for pos_weight, neg_weight in
                        zip(positive_weights.values(), negative_weights.values())]
        sample_weights = compute_sample_weight(class_weight=class_weight, y=train_tags)

        labels = np.delete(dataset_handler.mlb.classes_, 1)  # remove the not-ak label

        train_tags = df[labels].iloc[
            train_indices].values.astype('float32')
        val_tags = df[labels].iloc[
            val_indices].values.astype('float32')
        test_tags = df[labels].iloc[
            test_indices].values.astype('float32')

        model = EmailBERT().get_model()
        model.compile(loss=loss, optimizer=optimizer)

        model.fit(
            [train_ids, train_attention], train_tags, epochs=num_epochs,
            batch_size=batch_size,
            validation_data=([valid_ids, valid_attention], val_tags),
            class_weight=generate_class_weights(train_tags)
        )

        # model = tf.keras.models.load_model('bert', custom_objects={"TFDistilBertModel": TFDistilBertModel})
        test_output = model.predict([test_ids, test_attention])

        threshold = 0.5
        predicted_labels = (test_output >= threshold).astype(int)

        tags.append(test_tags)
        pred.append(predicted_labels)

        tags = np.array(list(chain.from_iterable(tags)))
        pred = np.array(list(chain.from_iterable(pred)))

        display_results(tags, pred)
        draw_matrix(tags, pred)

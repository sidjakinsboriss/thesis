from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast

from dl.classifiers.bert.bert import EmailBERT
from dl.constants import MAX_SEQUENCE_LENGTH, MODEL_NAME
from dl.dataset_handler import DatasetHandler
from dl.utils import display_results, draw_matrix, generate_class_weights


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

    # Specify hyper-parameters here
    batch_size = 64
    num_epochs = 40
    lr = 0.00001

    tags = []
    pred = []

    for train_indices, val_indices, test_indices in dataset_handler.get_indices(under_sample=True):
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

        train_texts = df['BODY'].iloc[train_indices].tolist()
        val_texts = df['BODY'].iloc[val_indices].tolist()
        test_texts = df['BODY'].iloc[test_indices].tolist()

        train_ids, train_attention = batch_encode(tokenizer, train_texts)
        valid_ids, valid_attention = batch_encode(tokenizer, val_texts)
        test_ids, test_attention = batch_encode(tokenizer, test_texts)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Training tags
        labels = np.delete(dataset_handler.mlb.classes_, 1)  # remove the not-ak label
        train_tags = df[labels].iloc[train_indices].values.astype('float32')
        val_tags = df[labels].iloc[val_indices].values.astype('float32')
        test_tags = df[labels].iloc[test_indices].values.astype('float32')

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
        test_output = tf.math.sigmoid(test_output).numpy()

        threshold = 0.5
        predicted_labels = (test_output >= threshold).astype(int)

        tags.append(test_tags)
        pred.append(predicted_labels)

        tags = np.array(list(chain.from_iterable(tags)))
        pred = np.array(list(chain.from_iterable(pred)))

        display_results(tags, pred)
        draw_matrix(tags, pred)

    # Save the model
    model.save('bert.h5')

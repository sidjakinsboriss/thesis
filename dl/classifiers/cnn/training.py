import math
import os
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf

from dl.classifiers.cnn.cnn import EmailCNN
from dl.constants import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
from dl.dataset_handler import DatasetHandler
from dl.utils import get_embedding_matrix, generate_class_weights, draw_matrix, display_results, \
    draw_class_confusion_matrices

if __name__ == '__main__':
    include_parent_email = False

    # df = pd.read_json(os.path.join(os.getcwd(), '../data/preprocessed.json'), orient='index')
    # df.to_csv(os.path.join(os.getcwd(), '../data/dataframe.csv'), index=None)

    df = pd.read_csv(os.path.join(os.getcwd(), '../data/dataframe.csv'))
    dataset_handler = DatasetHandler(df)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    emails = df['CONTENT'].tolist()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(emails)
    sequences = tokenizer.texts_to_sequences(emails)

    # Hyper-parameters
    batch_size = 32
    epochs = 15
    n_classes = 5
    lr = 0.0001
    hidden_size = 256

    word_index = tokenizer.word_index
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                                     maxlen=MAX_SEQUENCE_LENGTH)

    embedding_matrix = get_embedding_matrix(EMBEDDING_DIM, word_index, use_so=False)
    vocab_length = len(word_index) + 1

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5,
                                                  restore_best_weights=True)
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(patience=5)

    tags = []
    pred = []

    for train_indices, val_indices, test_indices in dataset_handler.get_indices():
        train_sequences = padded_sequences[train_indices]
        val_sequences = padded_sequences[val_indices]
        test_sequences = padded_sequences[test_indices]

        if include_parent_email:
            train_parent_indices = dataset_handler.get_parent_indices(train_indices)
            val_parent_indices = dataset_handler.get_parent_indices(val_indices)
            test_parent_indices = dataset_handler.get_parent_indices(test_indices)

            train_parent_sequences = padded_sequences[train_parent_indices]
            val_parent_sequences = padded_sequences[val_parent_indices]
            test_parent_sequences = padded_sequences[test_parent_indices]

            train_sequences = dataset_handler.concatenate_with_parent_indices(
                train_sequences, train_indices,
                train_parent_sequences, train_parent_indices)
            val_sequences = dataset_handler.concatenate_with_parent_indices(
                val_sequences, val_indices,
                val_parent_sequences, val_parent_indices)
            test_sequences = dataset_handler.concatenate_with_parent_indices(
                test_sequences, test_indices,
                test_parent_sequences, test_parent_indices)

        labels = np.delete(dataset_handler.mlb.classes_, 1)

        train_tags = df[labels].iloc[train_indices].values.astype('float32')
        val_tags = df[labels].iloc[val_indices].values.astype('float32')
        test_tags = df[labels].iloc[test_indices].values.astype('float32')

        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        model = EmailCNN(vocab_length, hidden_size, embedding_matrix).get_model()
        model.compile(optimizer=optimizer, loss=loss)

        steps_per_epoch = math.floor(len(train_sequences) / batch_size)
        steps_per_epoch_val = math.floor(len(val_sequences) / batch_size)

        model.fit(train_sequences, train_tags,
                  steps_per_epoch=steps_per_epoch, epochs=epochs,
                  validation_data=(val_sequences, val_tags),
                  batch_size=batch_size,
                  validation_steps=steps_per_epoch_val,
                  class_weight=generate_class_weights(train_tags))

        test_output = model.predict(test_sequences)
        test_output = tf.math.sigmoid(test_output).numpy()

        threshold = 0.5
        predicted_labels = (test_output >= threshold).astype(int)

        tags.append(test_tags)
        pred.append(predicted_labels)

        tags = np.array(list(chain.from_iterable(tags)))
        pred = np.array(list(chain.from_iterable(pred)))

        display_results(tags, pred)
        draw_class_confusion_matrices(tags, pred)
        draw_matrix(tags, pred)

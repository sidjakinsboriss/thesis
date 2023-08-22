import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf

from dl.classifiers.lstm.lstm import EmailLSTMHyperModel
from dl.constants import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
from dl.dataset_handler import DatasetHandler
from dl.utils import get_embedding_matrix, generate_class_weights

if __name__ == '__main__':
    df = pd.read_json('data/labeled_dataset_preprocessed.json', orient='index')
    dataset_handler = DatasetHandler(df)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    emails = df['BODY'].tolist()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(emails)
    sequences = tokenizer.texts_to_sequences(emails)

    word_index = tokenizer.word_index
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    embedding_matrix = get_embedding_matrix(EMBEDDING_DIM, word_index, use_so=False)
    vocab_length = len(word_index) + 1

    train_indices, val_indices, test_indices = dataset_handler.get_indices_for_optimization(under_sample=True)

    # Sequences
    train_sequences = padded_sequences[train_indices]
    val_sequences = padded_sequences[val_indices]
    test_sequences = padded_sequences[test_indices]

    # Tags
    labels = np.delete(dataset_handler.mlb.classes_, 1)  # remove the not-ak label
    train_tags = df[labels].iloc[train_indices].values.astype('float32')
    val_tags = df[labels].iloc[val_indices].values.astype('float32')
    test_tags = df[labels].iloc[test_indices].values.astype('float32')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    class_weights = generate_class_weights(train_tags)

    tuner = kt.Hyperband(hypermodel=EmailLSTMHyperModel(vocab_length, embedding_matrix),
                         objective='val_loss',
                         max_epochs=30,
                         factor=3,
                         directory='lstm_optimization',
                         project_name='lstm_optimization')
    tuner.search(train_sequences, train_tags,
                 validation_data=(val_sequences, val_tags), class_weight=class_weights, callbacks=[stop_early])

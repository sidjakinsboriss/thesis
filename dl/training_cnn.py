import os

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from dataset_handler import DatasetHandler
from dl.classifiers.cnn import EmailCNNHyperModel
from utils import get_embedding_matrix, generate_class_weights

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 100

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
    lr = 0.0001
    filters = 32
    hidden_layer_size = 64
    num_convolutions = 1

    word_index = tokenizer.word_index
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                                     maxlen=MAX_SEQUENCE_LENGTH)

    embedding_matrix = get_embedding_matrix(EMBEDDING_DIM, word_index,
                                            use_so=False)
    vocab_length = len(word_index) + 1

    tags = []
    pred = []

    train_indices, val_indices, test_indices = dataset_handler.get_indices()

    train_sequences = padded_sequences[train_indices]
    val_sequences = padded_sequences[val_indices]
    test_sequences = padded_sequences[test_indices]

    labels = dataset_handler.mlb.classes_

    train_tags = df[labels].iloc[train_indices].values.astype('float32')
    val_tags = df[labels].iloc[val_indices].values.astype('float32')
    test_tags = df[labels].iloc[test_indices].values.astype('float32')

    class_weights = generate_class_weights(train_tags)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner = kt.Hyperband(hypermodel=EmailCNNHyperModel(vocab_length, embedding_matrix),
                         objective='val_loss',
                         max_epochs=30,
                         factor=3,
                         directory='cnn_optimization',
                         project_name='cnn_optimization')
    tuner.search(train_sequences, train_tags,
                 validation_data=(val_sequences, val_tags), class_weight=class_weights, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
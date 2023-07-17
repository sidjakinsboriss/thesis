import math
import os
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf

from classifiers.rnn import EmailRNN
from dataset_handler import DatasetHandler
# from dl.loss import calculate_class_weights, pos_weight
# from dl.loss import calculate_class_weights
from utils import get_embedding_matrix, generate_class_weights, display_results, draw_matrix, \
    draw_class_confusion_matrices, calculate_pos_weight

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 100


def batch_generator(sequences, tags, batch_size, steps_per_epoch):
    while True:
        for i in range(steps_per_epoch):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_indices = sequences[start_idx:end_idx]
            batch_tags = tags[start_idx:end_idx]

            yield batch_indices, batch_tags


if __name__ == '__main__':
    include_parent_email = False

    # df = pd.read_json(os.path.join(os.getcwd(), '../data/preprocessed.json'), orient='index')
    # df.to_csv(os.path.join(os.getcwd(), '../data/dataframe.csv'), index=None)

    df = pd.read_csv(os.path.join(os.getcwd(), '../data/dataframe.csv'))
    dataset_handler = DatasetHandler(df)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()
    dataset_handler.add_powerlabels()

    pos_weight = calculate_pos_weight(df)

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
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    embedding_matrix = get_embedding_matrix(EMBEDDING_DIM, word_index, use_so=False)
    vocab_length = len(word_index) + 1

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(patience=5)

    tags = []
    pred = []

    for train_indices, val_indices, test_indices in dataset_handler.get_indices():
        train_sequences = padded_sequences[train_indices]
        val_sequences = padded_sequences[val_indices]
        test_sequences = padded_sequences[test_indices]

        train_parent_indices = dataset_handler.get_parent_indices(train_indices)
        val_parent_indices = dataset_handler.get_parent_indices(val_indices)
        test_parent_indices = dataset_handler.get_parent_indices(test_indices)

        train_parent_sequences = padded_sequences[train_parent_indices]
        val_parent_sequences = padded_sequences[val_parent_indices]
        test_parent_sequences = padded_sequences[test_parent_indices]

        train_sequences = dataset_handler.concatenate_with_parent_indices(train_sequences, train_indices,
                                                                          train_parent_sequences, train_parent_indices)
        val_sequences = dataset_handler.concatenate_with_parent_indices(val_sequences, val_indices,
                                                                        val_parent_sequences, val_parent_indices)
        test_sequences = dataset_handler.concatenate_with_parent_indices(test_sequences, test_indices,
                                                                         test_parent_sequences, test_parent_indices)

        train_tags = df[dataset_handler.mlb.classes_].iloc[train_indices].values.astype('float32')
        val_tags = df[dataset_handler.mlb.classes_].iloc[val_indices].values.astype('float32')
        test_tags = df[dataset_handler.mlb.classes_].iloc[test_indices].values.astype('float32')

        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
        model = EmailRNN(vocab_length, hidden_size, embedding_matrix,
                         optimizer).get_model()

        steps_per_epoch = math.floor(len(train_sequences) / batch_size)
        steps_per_epoch_val = math.floor(len(val_sequences) / batch_size)

        class_weights = generate_class_weights(train_tags)
        calculate_class_weights(df)

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

    # stop_early = EarlyStopping(monitor='val_loss', patience=5)
    # tuner = kt.Hyperband(hypermodel=EmailLSTMHyperModel(vocab_length, embedding_matrix),
    #                      objective='val_loss',
    #                      max_epochs=20,
    #                      factor=3,
    #                      directory='my_dir',
    #                      project_name='lstm_optimization')
    # tuner.search(train_sequences, train_tags,
    #              validation_data=(val_sequences, val_tags), class_weight=class_weights, callbacks=[stop_early])
    #
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    #
    # print(f"""
    # The hyperparameter search is complete. The optimal number of units in the first densely-connected
    # layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    # is {best_hps.get('learning_rate')}.
    # """)
    #
    # # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    # model = tuner.hypermodel.build(best_hps)
    # # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # history = model.fit(train_sequences, train_tags, epochs=epochs, batch_size=batch_size,
    #                     validation_data=(val_sequences, val_tags), class_weight=class_weights)
    #
    # val_acc_per_epoch = history.history['val_accuracy']
    # best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    # print('Best epoch: %d' % (best_epoch,))

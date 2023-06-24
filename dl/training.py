import os

import keras_tuner as kt
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, plot_model

from classifiers.rnn import EmailRNN, EmailLSTMHyperModel
from dataset_split_handler import DatasetHandler
from utils import get_embedding_matrix, generate_class_weights

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 100

if __name__ == '__main__':
    include_parent_email = False

    df = pd.read_csv(os.path.join(os.getcwd(), '../data/dataframe.csv'))
    dataset_handler = DatasetHandler(df, None, include_parent_email)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    emails = df['CONTENT'].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(emails)
    sequences = tokenizer.texts_to_sequences(emails)

    word_index = tokenizer.word_index
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    embedding_matrix = get_embedding_matrix(EMBEDDING_DIM, word_index)

    train_indices, val_indices, test_indices = dataset_handler.get_indices()

    train_sequences = padded_sequences[train_indices]
    val_sequences = padded_sequences[val_indices]
    test_sequences = padded_sequences[test_indices]

    train_tags = df[dataset_handler.mlb.classes_].iloc[train_indices].values.astype('float32')
    val_tags = df[dataset_handler.mlb.classes_].iloc[val_indices].values.astype('float32')
    test_tags = df[dataset_handler.mlb.classes_].iloc[test_indices].values.astype('float32')

    # Hyper-parameters
    batch_size = 32
    epochs = 10
    n_classes = 5
    lr = 0.0001
    hidden_size = 128

    optimizer = Adam(learning_rate=lr)
    vocab_length = len(word_index) + 1

    class_weights = generate_class_weights(train_tags)

    model = EmailRNN(vocab_length, hidden_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, embedding_matrix,
                     optimizer).get_model()

    # model.fit(train_sequences, train_tags, epochs=epochs, batch_size=batch_size,
    #           validation_data=(val_sequences, val_tags), class_weight=class_weights)
    #
    # test_output = model.predict(test_sequences)
    # threshold = 0.5
    # predicted_labels = (test_output >= threshold).astype(np.int)
    #
    # display_results(test_tags, predicted_labels)
    # draw_matrix(test_tags, predicted_labels)

    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    tuner = kt.Hyperband(hypermodel=EmailLSTMHyperModel(vocab_length, embedding_matrix),
                         objective='val_loss',
                         max_epochs=20,
                         factor=3,
                         directory='my_dir',
                         project_name='lstm_optimization')
    tuner.search(train_sequences, train_tags,
                 validation_data=(val_sequences, val_tags), class_weight=class_weights, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    history = model.fit(train_sequences, train_tags, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_sequences, val_tags), class_weight=class_weights)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

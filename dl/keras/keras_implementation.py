import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.optimizers import Adam
from sklearn.utils import class_weight

from dl.dataset_split_handler import DatasetHandler
from dl.training import get_word_embeddings

if __name__ == '__main__':
    word_embeddings = get_word_embeddings(os.path.join(os.getcwd(), '../word2vec_model'),
                                          use_trained_weights=True)
    include_parent_email = False

    df = pd.read_csv(os.path.join(os.getcwd(), '../../data/dataframe.csv'))
    dataset_handler = DatasetHandler(df, word_embeddings.key_to_index, include_parent_email)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    gensim_model = Word2Vec.load('../word2vec_model')

    emails = df['CONTENT'].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(emails)
    sequences = tokenizer.texts_to_sequences(emails)

    word_index = tokenizer.word_index
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    embedding_dim = 200
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in gensim_model.wv:
            embedding_matrix[i] = gensim_model.wv[word]

    train_indices, val_indices, test_indices = dataset_handler.get_data()

    train_sequences = padded_sequences[train_indices]
    val_sequences = padded_sequences[val_indices]
    test_sequences = padded_sequences[test_indices]

    train_tags = df[dataset_handler.mlb.classes_].iloc[train_indices].values
    val_tags = df[dataset_handler.mlb.classes_].iloc[val_indices].values
    test_tags = df[dataset_handler.mlb.classes_].iloc[test_indices].values

    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, input_length=max_sequence_length, weights=[embedding_matrix],
                  trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(5, activation='sigmoid'))

    batch_size = 32
    epochs = 10
    n_classes = 5

    optimizer = Adam(learning_rate=0.001)

    class_count = [0] * n_classes
    for classes in train_tags:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1

    # Compute class weights using balanced method
    class_weights = [len(padded_sequences) / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_weight_dict = dict(enumerate(class_weights))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_sequences, train_tags, epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict)

    test_loss, test_accuracy = model.evaluate(test_sequences, test_tags)
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

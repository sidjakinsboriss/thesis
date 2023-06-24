import keras_tuner
from keras.initializers.initializers import Constant
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC
from keras.models import Sequential
from keras.optimizers import Adam

NUM_CLASSES = 5
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 100


class EmailRNN:
    def __init__(self, vocab_length, hidden_size, embedding_dim, max_sequence_length, embedding_matrix, optimizer):
        self.model = Sequential()
        self.model.add(
            Embedding(vocab_length, embedding_dim, input_length=max_sequence_length,
                      embeddings_initializer=Constant(embedding_matrix),
                      trainable=False, mask_zero=True))
        self.model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(hidden_size)))
        self.model.add(Dense(NUM_CLASSES))

        metrics = [AUC(name="average_precision", curve="PR", multi_label=True)]

        self.model.compile(loss=BinaryCrossentropy(from_logits=True),
                           optimizer=optimizer, metrics=metrics)

    def get_model(self):
        return self.model


class EmailLSTMHyperModel(keras_tuner.HyperModel):
    def __init__(self, vocab_length, embedding_matrix):
        super().__init__()
        self.vocab_length = vocab_length
        self.embedding_matrix = embedding_matrix

    def build(self, hp):
        model = Sequential()
        model.add(
            Embedding(self.vocab_length, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,
                      embeddings_initializer=Constant(self.embedding_matrix),
                      trainable=False, mask_zero=True))

        hp_hidden_size = hp.Int('hidden_size', min_value=32, max_value=512, step=32)

        hp_num_layers = hp.Int('num_layers', min_value=0, max_value=3, step=1)

        for i in range(hp_num_layers):
            model.add(Bidirectional(LSTM(hp_hidden_size, return_sequences=True)))

        model.add(Bidirectional(LSTM(hp_hidden_size)))
        model.add(Dense(NUM_CLASSES))

        metrics = [AUC(name="average_precision", curve="PR", multi_label=True)]

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        optimizer = Adam(learning_rate=hp_learning_rate)

        model.compile(loss=BinaryCrossentropy(from_logits=True),
                      optimizer=optimizer, metrics=metrics)

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', min_value=16, max_value=128, step=32),
            **kwargs,
        )

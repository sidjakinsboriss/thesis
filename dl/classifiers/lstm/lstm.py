import keras_tuner as kt
import tensorflow as tf

from dl.constants import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, NUM_CLASSES


class EmailRNN:
    def __init__(self, vocab_length, embedding_matrix, lstm_size=512, hidden_size=128, num_hidden_layers=0):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Embedding(vocab_length, EMBEDDING_DIM,
                                      input_length=MAX_SEQUENCE_LENGTH,
                                      embeddings_initializer=tf.keras.initializers.Constant(
                                          embedding_matrix),
                                      trainable=True, mask_zero=True, batch_size=32))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size)))

        for i in range(num_hidden_layers):
            model.add(tf.keras.layers.Dense(hidden_size))

        model.add(tf.keras.layers.Dense(NUM_CLASSES))

        self.model = model

    def get_model(self):
        return self.model


class EmailLSTMHyperModel(kt.HyperModel):
    def __init__(self, vocab_length, embedding_matrix):
        super().__init__()
        self.vocab_length = vocab_length
        self.embedding_matrix = embedding_matrix

    def build(self, hp):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Embedding(self.vocab_length, EMBEDDING_DIM,
                                      input_length=MAX_SEQUENCE_LENGTH,
                                      embeddings_initializer=
                                      tf.keras.initializers.Constant(
                                          self.embedding_matrix
                                      ),
                                      trainable=True, mask_zero=True))

        hp_lstm_size = hp.Choice('lstm_size', values=[64, 128, 256, 512])
        hp_hidden_size = hp.Int('hidden_size', min_value=32, max_value=512,
                                step=32)

        hp_num_layers = hp.Int('num_layers', min_value=0, max_value=3, step=1)

        model.add(
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp_lstm_size))
        )

        for i in range(hp_num_layers):
            model.add(tf.keras.layers.Dense(hp_hidden_size))

        model.add(tf.keras.layers.Dense(NUM_CLASSES))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        model.compile(loss=loss, optimizer=optimizer)

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', min_value=32, max_value=64,
                              step=32),
            **kwargs,
        )

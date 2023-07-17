import keras_tuner as kt
import tensorflow as tf

NUM_CLASSES = 5
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 100


class EmailRNN:
    def __init__(self, vocab_length, hidden_size, embedding_matrix, optimizer):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Embedding(vocab_length, EMBEDDING_DIM,
                                      input_length=MAX_SEQUENCE_LENGTH * 2,
                                      embeddings_initializer=tf.keras.initializers.Constant(
                                          embedding_matrix),
                                      trainable=True, mask_zero=True))
        model.add(
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size)))

        model.add(tf.keras.layers.Dense(NUM_CLASSES))

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=optimizer)

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

        hp_hidden_size = hp.Int('hidden_size', min_value=32, max_value=512,
                                step=32)

        hp_num_layers = hp.Int('num_layers', min_value=0, max_value=3, step=1)

        for i in range(hp_num_layers):
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hp_hidden_size, return_sequences=True)))

        model.add(
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp_hidden_size))
        )
        model.add(tf.keras.layers.Dense(NUM_CLASSES))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        optimizer = tf.keras.optimizers.AdamW(learning_rate=hp_learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        model.compile(loss=loss, optimizer=optimizer)

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', min_value=16, max_value=128,
                              step=32),
            **kwargs,
        )

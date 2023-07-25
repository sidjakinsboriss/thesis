import keras_tuner as kt
import tensorflow as tf

from dl.constants import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, NUM_CLASSES


class EmailCNN:
    def __init__(self, vocab_length, embedding_matrix, filters=32, hidden_layer_size=32, num_convolutions=1):
        input_layer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,))
        embedding = tf.keras.layers.Embedding(vocab_length, EMBEDDING_DIM,
                                              input_length=MAX_SEQUENCE_LENGTH,
                                              embeddings_initializer=tf.keras.initializers.Constant(
                                                  embedding_matrix),
                                              trainable=True, mask_zero=True)(input_layer)

        kernel_sizes = [8]
        conv_layers = [
            tf.keras.layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size)(embedding)
            for kernel_size in kernel_sizes
        ]

        pooling_layers = [
            tf.keras.layers.GlobalMaxPooling1D()(conv)
            for conv in conv_layers
        ]

        if len(pooling_layers) == 1:
            concatenated = pooling_layers[0]
        else:
            concatenated = tf.keras.layers.concatenate(pooling_layers, axis=1)

        hidden = tf.keras.layers.Flatten()(concatenated)

        if hidden_layer_size > 0:
            hidden = tf.keras.layers.Dense(hidden_layer_size, activation='relu')(hidden)

        outputs = tf.keras.layers.Dense(NUM_CLASSES)(hidden)
        self.model = tf.keras.Model(inputs=[embedding], outputs=outputs)

    def get_model(self):
        return self.model


class EmailCNNHyperModel(kt.HyperModel):
    def __init__(self, vocab_length, embedding_matrix):
        super().__init__()
        self.vocab_length = vocab_length
        self.embedding_matrix = embedding_matrix

    def build(self, hp):
        hp_num_filters = hp.Choice('num_filters', values=[32, 64])
        hp_num_conv_layers = hp.Choice('num_layers', values=[1, 2, 3])
        hp_kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])
        hp_hidden_size = hp.Choice('hidden_size', values=[0, 64, 128])
        hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])

        input_layer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,))
        embedding = tf.keras.layers.Embedding(self.vocab_length, EMBEDDING_DIM,
                                              input_length=MAX_SEQUENCE_LENGTH,
                                              embeddings_initializer=tf.keras.initializers.Constant(
                                                  self.embedding_matrix),
                                              trainable=True, mask_zero=True)(input_layer)

        conv_layers = [
            tf.keras.layers.Conv1D(filters=hp_num_filters,
                                   kernel_size=hp_kernel_size)(embedding)
            for _ in range(hp_num_conv_layers)
        ]
        pooling_layers = [
            tf.keras.layers.GlobalMaxPooling1D()(conv)
            for conv in conv_layers
        ]

        if len(pooling_layers) == 1:
            concatenated = pooling_layers[0]
        else:
            concatenated = tf.keras.layers.concatenate(pooling_layers, axis=1)

        hidden = tf.keras.layers.Flatten()(concatenated)
        if hp_hidden_size > 0:
            hidden = tf.keras.layers.Dense(hp_hidden_size, activation='relu')(hidden)

        outputs = tf.keras.layers.Dense(NUM_CLASSES)(hidden)

        optimizer = tf.keras.optimizers.AdamW(learning_rate=hp_learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        model = tf.keras.Model(inputs=[input_layer], outputs=outputs)
        model.compile(loss=loss, optimizer=optimizer)

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', min_value=32, max_value=128, step=32),
            **kwargs,
        )

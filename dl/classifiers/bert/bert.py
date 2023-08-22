import keras_tuner as kt
import tensorflow as tf
from transformers import TFDistilBertModel

from dl.constants import MODEL_NAME, NUM_CLASSES, MAX_SEQUENCE_LENGTH


class EmailBERT:
    def __init__(self):
        bert = TFDistilBertModel.from_pretrained(MODEL_NAME)

        # Input layers
        input_ids_layer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,),
                                                name='input_ids',
                                                dtype='int32')
        input_attention_layer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,),
                                                      name='input_attention',
                                                      dtype='int32')

        # DistilBERT outputs a tuple where the first element at index 0
        # represents the hidden-state at the output of the model's last layer.
        # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
        last_hidden_state = bert([input_ids_layer, input_attention_layer])[0]

        # We only care about DistilBERT's output for the [CLS] token,
        # which is located at index 0 of every encoded sequence.
        cls_token = last_hidden_state[:, 0, :]

        output = tf.keras.layers.Dense(NUM_CLASSES)(cls_token)
        model = tf.keras.models.Model([input_ids_layer, input_attention_layer], output)

        self.model = model

    def get_model(self):
        return self.model


class EmailBERTHyperModel(kt.HyperModel):
    def __init__(self):
        super().__init__()

    def build(self, hp):
        hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])

        bert = TFDistilBertModel.from_pretrained(MODEL_NAME)

        input_ids_layer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,),
                                                name='input_ids',
                                                dtype='int32')
        input_attention_layer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,),
                                                      name='input_attention',
                                                      dtype='int32')

        last_hidden_state = bert([input_ids_layer, input_attention_layer])[0]
        cls_token = last_hidden_state[:, 0, :]

        output = tf.keras.layers.Dense(NUM_CLASSES)(cls_token)

        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        model = tf.keras.models.Model([input_ids_layer, input_attention_layer], output)
        model.compile(loss=loss, optimizer=optimizer)

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', min_value=32, max_value=64, step=32),
            **kwargs,
        )

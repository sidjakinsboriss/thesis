import tensorflow as tf
from transformers import TFDistilBertModel

from dl.loss import weighted_loss

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 100
RANDOM_STATE = 42
NUM_CLASSES = 5


class BertMultiLabel:
    def __init__(self, optimizer):
        self.bert = TFDistilBertModel.from_pretrained(MODEL_NAME)
        self.optimizer = optimizer

        weight_initializer = tf.keras.initializers.GlorotNormal(seed=RANDOM_STATE)

        # Define input layers
        input_ids_layer = tf.keras.layers.Input(shape=(MAX_LENGTH * 2,),
                                                name='input_ids',
                                                dtype='int32',
                                                batch_size=32)
        input_attention_layer = tf.keras.layers.Input(shape=(MAX_LENGTH * 2,),
                                                      name='input_attention',
                                                      dtype='int32',
                                                      batch_size=32)

        # DistilBERT outputs a tuple where the first element at index 0
        # represents the hidden-state at the output of the model's last layer.
        # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
        last_hidden_state = self.bert([input_ids_layer, input_attention_layer])[0]

        # We only care about DistilBERT's output for the [CLS] token,
        # which is located at index 0 of every encoded sequence.
        # Splicing out the [CLS] tokens gives us 2D data.
        cls_token = last_hidden_state[:, 0, :]

        # Define a single node that makes up the output layer (for binary classification)
        output = tf.keras.layers.Dense(NUM_CLASSES,
                                       kernel_initializer=weight_initializer,
                                       kernel_constraint=None,
                                       bias_initializer='zeros',
                                       activation='sigmoid'
                                       )(cls_token)

        # Define the model
        model = tf.keras.models.Model([input_ids_layer, input_attention_layer], output)

        # Compile the model
        model.compile(optimizer=optimizer,
                      loss=weighted_loss)

        self.model = model

    def get_model(self):
        return self.model

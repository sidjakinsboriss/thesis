from keras import Input, Model
from keras.initializers.initializers import GlorotNormal
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC
from keras.optimizers import Adam
from transformers import TFDistilBertModel

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 200
LAYER_DROPOUT = 0.2
RANDOM_STATE = 42
NUM_CLASSES = 5


class BertMultiLabel:
    def __init__(self):
        self.bert = TFDistilBertModel.from_pretrained(MODEL_NAME)

        for layer in self.bert.layers:
            layer.trainable = False

    def get_model(self):
        metrics = [AUC(name="average_precision", curve="PR", multi_label=True)]

        optimizer = Adam(learning_rate=1e-5, epsilon=1e-08)

        weight_initializer = GlorotNormal(seed=RANDOM_STATE)

        # Define input layers
        input_ids_layer = Input(shape=(MAX_LENGTH,),
                                name='input_ids',
                                dtype='int32')
        input_attention_layer = Input(shape=(MAX_LENGTH,),
                                      name='input_attention',
                                      dtype='int32')

        # DistilBERT outputs a tuple where the first element at index 0
        # represents the hidden-state at the output of the model's last layer.
        # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
        last_hidden_state = self.bert([input_ids_layer, input_attention_layer])[0]

        # We only care about DistilBERT's output for the [CLS] token,
        # which is located at index 0 of every encoded sequence.
        # Splicing out the [CLS] tokens gives us 2D data.
        cls_token = last_hidden_state[:, 0, :]

        # Define a single node that makes up the output layer (for binary classification)
        output = Dense(NUM_CLASSES,
                       kernel_initializer=weight_initializer,
                       kernel_constraint=None,
                       bias_initializer='zeros'
                       )(cls_token)

        # Define the model
        model = Model([input_ids_layer, input_attention_layer], output)

        # Compile the model
        model.compile(optimizer=optimizer,
                      loss=BinaryCrossentropy(),
                      metrics=metrics)

        return model

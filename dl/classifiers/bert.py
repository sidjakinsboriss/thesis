from keras.initializers.initializers import TruncatedNormal
from keras.layers import Input, Dropout, Dense
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC
from keras.models import Model
from keras.optimizers import Adam
from transformers import TFDistilBertForSequenceClassification, \
    DistilBertConfig

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 200


class BertMultiLabel:
    def __init__(self):
        config = DistilBertConfig.from_pretrained(MODEL_NAME)

        transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, output_hidden_states=False, num_labels=5
        )

        bert = transformer_model.layers[0]

        # The input is a dictionary of word identifiers
        input_ids = Input(shape=(MAX_LENGTH,), name='input_ids', dtype='int32')
        inputs = {'input_ids': input_ids}

        # Here we select the representation of the first token ([CLS]) for classification
        # (a.k.a. "pooled representation")
        bert_model = bert(inputs)[0][:, 0, :]

        # Add a dropout layer and the output layer
        dropout = Dropout(config.dropout, name='pooled_output')
        pooled_output = dropout(bert_model, training=False)
        output = Dense(
            units=5,
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            name='output'
        )(pooled_output)

        model = Model(inputs=inputs, outputs=output, name='BERT_MultiLabel')

        loss = BinaryCrossentropy(from_logits=True)
        optimizer = Adam(5e-5)
        metrics = [
            "binary_accuracy",
            AUC(name="average_precision", curve="PR", multi_label=True)
        ]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model

    def get_model(self):
        return self.model

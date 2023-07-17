import numpy as np
import tensorflow as tf


def calculate_class_weights(df):
    N = len(df)
    labels = df.iloc[:, 4:9]

    positive_weights = {}
    negative_weights = {}

    for label in sorted(labels):
        positive_weights[label] = N / (2 * sum(df[label] == 1))
        negative_weights[label] = N / (2 * sum(df[label] == 0))

    global Wp
    global Wn

    Wp = positive_weights
    Wn = negative_weights


def weighted_loss(y_true, y_logit):
    '''
    Multi-label cross-entropy
    * Required "Wp", "Wn" as positive & negative class-weights
    y_true: true value
    y_logit: predicted value
    '''
    losses = []
    for k in range(32):
        true = y_true[k]
        logit = y_logit[k]

        loss = float(0)

        for i, key in enumerate(Wp.keys()):
            first_term = Wp[key] * true[i] * tf.keras.backend.log(logit[i] + tf.keras.backend.epsilon())
            second_term = Wn[key] * (1 - true[i]) * tf.keras.backend.log(
                1 - logit[i] + tf.keras.backend.epsilon())
            loss -= (first_term + second_term)

        losses.append(loss)

    return tf.reduce_mean(losses)


def pos_weight(df):
    global weights

    N = len(df)
    labels = df.iloc[:, 4:9]

    weights = []

    for label in sorted(labels):
        positive_count = sum(df[label] == 1)
        negative_count = N - positive_count
        weights.append(negative_count / positive_count)

    weights = np.array(weights)


def weighted_cross_entropy_loss(targets, logits):
    weighted_losses = tf.nn.weighted_cross_entropy_with_logits(targets, logits, weights)

    return tf.reduce_mean(weighted_losses)

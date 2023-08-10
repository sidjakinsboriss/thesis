import numpy as np
import tensorflow as tf


def calculate_pos_weight(labels):
    pos_weight = []
    for i in range(5):
        neg_count = len([label[i] for label in labels if label[i] == 0])
        pos_weight.append(neg_count / (len(labels) - neg_count))

    return np.array(pos_weight)


def custom_loss(pos_weights):
    def weighted_cross_entropy_loss(targets, logits):
        return tf.nn.weighted_cross_entropy_with_logits(targets, logits,
                                                        pos_weights)

    return weighted_cross_entropy_loss

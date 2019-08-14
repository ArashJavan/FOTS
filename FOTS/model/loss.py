import tensorflow as tf
from tensorflow import keras

class DetectionLoss(keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super(DetectionLoss, self).__init__(*args, **kwargs)

    def dice_coefficient(self, y_true_cls, y_pred_cls,
                     training_mask):
        """
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        """
        eps = 1e-5
        intersection = tf.c(y_true_cls * y_pred_cls * training_mask)
        union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def call(self, y_true, y_pred):
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred, mask = tf.split(y_true, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(y_pred, 1, 1)

        dice_cls_loss = self.dice_coefficient(y_true, y_pred)
        dice_cls_loss *= 0.01

import math
from tensorflow import keras


class TextDetector(keras.Model):
    """
    This is the detector-branch of the netwrok,
    we use as in EAST sigmoid activations, since the output is normalized , out \in [0, 1]
    """
    def __init__(self, shared_features, input_shape, activation="sigmoid", **kwargs):
        super(TextDetector, self).__init__(**kwargs)

        self.text_scale = input_shape[0]
        self.shared_featured = shared_features

        self.score_conv = keras.layers.Conv2D(filters=1, kernel_size=1, padding="same", activation=activation)
        self.geo_conv = keras.layers.Conv2D(filters=4, kernel_size=1, padding="same", activation=activation)
        self.angle_conv = keras.layers.Conv2D(filters=1, kernel_size=1, padding="same", activation=activation)

    def call(self, inputs, training=None, mask=None):
        outputs = self.shared_featured(inputs)
        score = self.score_conv(outputs)
        geo_map = self.geo_conv(outputs) * self.text_scale
        angle_map = (self.angle_conv(outputs) - 0.5) * math.pi  # theta \in [-90, 90]
        geometry = keras.layers.concatenate([geo_map, angle_map])
        return score, geometry

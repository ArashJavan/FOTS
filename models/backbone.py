import math
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from cytoolz.itertoolz import _outer_join
from dask.bag.core import groupby_tasks
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets


class SharedConvlution(keras.Model):
    num_features = [128, 64, 32]

    def __init__(self, backbone, bn=True, activation=None, **kwargs):
        super(SharedConvlution, self).__init__(**kwargs)
        self.backbone = backbone

        self.num_bb_layers = len(self.backbone.layers)

        self.g_s = []
        self.h_s = []
        for i in range(self.num_bb_layers):
            if i == 0:
                h = IdentityLayer()
            else:
                filters = self.get_filter_dim(i - 1)
                h = HLayer(filters, bn=bn, activation=activation)

            if i == self.num_bb_layers - 1:
                g = keras.layers.Conv2D(32, kernel_size=3, padding="same")
            else:
                g = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
            self.g_s.append(g)
            self.h_s.append(h)

    def get_filter_dim(self, i):
        if i < len(self.num_features) - 1:
            return self.num_features[i]
        else:
            return 32

    def call(self, inputs, training=None, mask=None):
        bb_outputs = self.backbone(inputs)

        outputs = None

        # We start from the last layer of our backbone model,
        # mostly the one with most number of feature maps
        # the so called top-down pathway and lateral connection
        # Feature Pyramid Networks for Object Detection
        # https://arxiv.org/abs/1612.03144
        for i in range(self.num_bb_layers):
            outputs = self.h_s[i]([outputs, bb_outputs[(self.num_bb_layers -1) - i]])
            outputs = self.g_s[i](outputs)
        return outputs


class IdentityLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(IdentityLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        :param inputs: a tuple from prev layers g_prev and current f-layer
        :param kwargs:
        :return:
        """
        return inputs[-1]

class HLayer(keras.layers.Layer):
    """
    For more info about h- and  g-layers see
    EAST: An Efficient and Accurate Scene Text Detector
    https://arxiv.org/abs/1704.03155

    The Hlayer performs basically following operations
    1. Cocanating input and output along the channel axis
    2. Dimension reduction via a 1x1 convolution
    3. fixed channel 3x3 conv, also a 3x3-Kernel ensures a noise reduction
    """
    def __init__(self, filters, bn=False, activation=None):
        super(HLayer, self).__init__()
        self.bn = bn
        self.activation = activation

        self.conv1x1 = keras.layers.Conv2D(filters, kernel_size=1, padding="same")
        self.conv3x3 = keras.layers.Conv2D(filters, kernel_size=3, padding="same")

        if bn:
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()

        if activation is not None:
            self.activation_1= keras.activations.get(activation)
            self.activation_2 = keras.activations.get(activation)

    def call(self, inputs, **kwargs):
        """
        :param inputs: a tuple from prev layers g_prev and current f-layer
        :param kwargs:
        :return:
        """
        output = keras.layers.concatenate(inputs, axis=-1)
        output = self.conv1x1(output)

        if self.bn:
            output = self.bn1(output)

        if self.activation is not None:
            output = self.activation_1(output)

        output = self.conv3x3(output)

        if self.bn:
            output = self.bn2(output)

        if self.activation is not None:
            output = self.activation_2(output)

        return output


class Resnet50Wrapper():
    layers_out = ["activation_9", "activation_21", "activation_39", "activation_48"]

    def __init__(self, input_shape, **kwargs):
        self._bb_model = keras.applications.resnet50.ResNet50(input_shape=input_shape, include_top=False)
        self._bb_layers = [self._bb_model.get_layer(layer_name) for layer_name in self.layers_out]

        bb_ouputs = [layer.output for layer in self._bb_layers]
        self.model = keras.Model(inputs=self._bb_model.input, outputs=bb_ouputs)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    @property
    def layers(self):
        return self._bb_layers

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


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
        self.angle_conv = keras.layers.Conv2D(filters=4, kernel_size=1, padding="same", activation=activation)

    def call(self, inputs, training=None, mask=None):
        self.text_scale = self.input_shape[0]

        outputs = self.shared_featured(inputs)
        score_map = self.score_conv(outputs)
        geo_map = self.geo_conv(outputs) * self.text_scale
        angle_map = (self.angle_conv(outputs) - 0.5) * math.pi

        return


input_shape = (640, 640, 3)
image_pathes = ["dog.jpg", "dog2.jpg"]
images = []
for img in image_pathes:
    img_raw = tf.io.read_file(img)
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [640, 640])
    img_final = img_final / 255.0
    images.append(img_final)
images = tf.convert_to_tensor(images)
# imgs = np.random.randn(2, 640, 640, 3).astype(np.float32)

resnet50_wrapper = Resnet50Wrapper(input_shape=input_shape)
shared_features = SharedConvlution(backbone=resnet50_wrapper)
detector = TextDetector(shared_features=shared_features, input_shape=input_shape)
Y = detector(images)

fig, axs = plt.subplots(2, 2)
axs = axs.flatten()
for i in range(4):
    img = tf.squeeze(Y[i])
    axs[i].imshow(img[1, :, :])
    axs[i].set_title(i)
fig.tight_layout()
plt.show()

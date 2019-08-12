import tensorflow as tf
from tensorflow import  keras
import tensorflow_datasets as tfds

# tfds works in both Eager and Graph modes


def conv3x3(filters, kernl_size=3, strides=1, padding='same'):
    """3x3 convolution with padding"""
    return keras.layers.Conv2D(filters, kernel_size=kernl_size, strides=strides, padding=padding, use_bias=False, kernel_initializer=tf.random_normal_initializer())

def conv1x1(filters, strides=1, padding="same"):
    """1x1 convolution"""
    return keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding)


class Bottleneck(keras.layers.Layer):
    expansion_factor = 4

    def __init__(self, in_filters, out_filters, activation="relu"):
        super(Bottleneck, self).__init__(name="resnet_bottleneck")

        self.downsample = None
        strides = 1

        # initializing internal vars
        self.activation = keras.activations.get(activation)

        # we shoudl downsample and expand the dimension, since in- and output channels mismatch
        if in_filters != (self.expansion_factor * out_filters) and in_filters != out_filters:
            self.downsample = keras.models.Sequential([
                conv1x1(out_filters, strides=2),
                keras.layers.BatchNormalization()
            ])
            strides = 2

        # building the netowrk
        filters = int(out_filters / self.expansion_factor)
        self.conv1 = conv1x1(filters, strides=strides)
        self.bn1 = keras.layers.BatchNormalization()

        self.conv2 = conv3x3(filters)
        self.bn2 = keras.layers.BatchNormalization()

        self.conv3 = conv1x1(out_filters)
        self.bn3 = keras.layers.BatchNormalization()
        self.fn1 = self.activation

    def call(self, inputs, **kwargs):
        identity = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.fn1(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.activation(out)
        return out


class BasicBlock(keras.layers.Layer):
    expansion_factor = 1
    def __init__(self, in_filters, out_filters, activation="relu"):
        super(BasicBlock, self).__init__(name="resnet_basicblock")

        self.downsample = None
        strides = 1

        # initializing internal vars
        self.activation = keras.activations.get(activation)

        # we shoudl downsample and expand the dimension, since in- and output channels mismatch
        if in_filters != out_filters:
            self.downsample = keras.models.Sequential([
                conv1x1(out_filters, strides=2),
                keras.layers.BatchNormalization()
            ])
            strides = 2

        # building the netowrk
        self.conv1 = conv3x3(out_filters, strides=strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.fn1 = self.activation
        self.conv2 = conv3x3(out_filters)
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        identity = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.fn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.activation(out)
        return out


# class ResNet(keras.Model):
#     expansion_factor = 2
#     def __init__(self, block, layers, in_filters=64, num_classes=1000, zero_init_residual=False, norm_layer=None, activation="relu", **kwargs):
#         super(ResNet, self).__init__(**kwargs)
#
#         if len(layers) != 4:
#             raise ValueError("ResNet is defined over 4 Layer blocks and not {}".format(len(layers)))
#
#         if norm_layer is None:
#             norm_layer = keras.layers.BatchNormalization
#         self._norm_layer = norm_layer
#
#         self.blocks = keras.model.Sequential(name='dynamic-blocks')
#
#         self._activation = keras.activations.get(activation)
#
#         self.in_filters = in_filters
#         self.out_filter = self.in_filters
#
#         # In the paper ResNet stride is set to =2 for the first conv-layer without dim. reduction,
#         # but keras does not support user defined padding to ge the same size
#         self.conv1 = conv3x3(self.in_filters, kernl_size=7, strides=2)  # out_shape = 1/2 in_shape
#         self.bn1 = self._norm_layer()
#         self.fn1 = self._activation
#         self.maxpool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2) # out_shape = 1/4 in_shape
#
#         out_filters = in_filters
#         self.layer1 = self._make_block_layer_(block, in_filters, out_filters, layers[0])
#
#         in_filters = out_filters
#         out_filters = self.expansion_factor * in_filters
#         self.layer2 = self._make_block_layer_(block, in_filters, out_filters, layers[1])
#
#         in_filters = out_filters
#         out_filters = self.expansion_factor * in_filters
#         self.layer3 = self._make_block_layer_(block, in_filters, out_filters, layers[2])
#
#         in_filters = out_filters
#         out_filters = self.expansion_factor * in_filters
#         self.layer4 = self._make_block_layer_(block, in_filters, out_filters, layers[3])
#
#         # self._make_alyer(block, layers)
#         self.avg_pool = keras.layers.GlobalAveragePooling2D()
#         self.fc = keras.layers.Dense(num_classes)
#
#     def call(self, inputs, training=None, mask=None):
#         out = self.conv1(inputs)
#         out = self.bn1(out)
#         out = self.fn1(out)
#         out = self.maxpool(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avg_pool(out)
#         out = self.fc(out)
#         return out
#
#     def _make_block_layer_(self, block, in_filters, out_filters, num_layers):
#         layer_block = keras.model.Sequential(name='dynamic-blocks')
#         for i in range(num_layers):
#             layer_block.add(block(in_filters, out_filters))
#             in_filters = out_filters
#         return layer_block
#
#     def _make_alyer(self, block, layers):
#         shortcut_dim_change = False
#         for blk_num in range(len(layers)):
#             for layer_num in range(layers[blk_num]):
#
#                 if blk_num != 0 and layer_num == 0:
#                     blk = block(self.out_filter, shortcut_dim_change=True, strides=2)
#                 else:
#                     if self.in_filters != self.out_filter:
#                         shortcut_dim_change = True
#                     else:
#                         shortcut_dim_change = False
#                     blk = block(self.out_filter, shortcut_dim_change=shortcut_dim_change)
#                 self.blocks.add(blk)
#                 self.in_filters = self.out_filter
#             self.out_filter *= 2
#

class ResNet(keras.Model):
    def __init__(self, res_block_arch, conv1_max=True, conv1_filters=64, conv1_kernel_size=7, conv1_strides=2, num_classes=1000, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.in_filters = conv1_filters

        # building conv1-block
        self.conv1_block = keras.models.Sequential(name='dynamic-blocks')
        self.conv1_block.add(conv3x3(conv1_filters, kernl_size=conv1_kernel_size, strides=conv1_strides)) # out_shape = 1/2 in_shape
        self.conv1_block.add(keras.layers.BatchNormalization())
        self.conv1_block.add(keras.layers.ReLU())

        if conv1_max:
            self.conv1_block.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)) # out_shape = 1/4 in_shape

        # self.blocks = keras.model.Sequential(name='dynamic-blocks')
        self.blocks = []
        for block_arch in res_block_arch:
            blk = self._make_block(block_arch)
            self.blocks.append(blk)

            # self._make_alyer(block, layers)
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        out = self.conv1_block(inputs)
        for blk in self.blocks:
            out = blk(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out

    def _make_block(self, block_arch):
        block_builder = block_arch.block
        in_filters = self.in_filters
        out_filters = block_arch.filters * block_builder.expansion_factor

        layer_X = keras.models.Sequential(name='dynamic-blocks')

        for i in range(1, block_arch.num_layers):
            layer_X.add(block_builder(in_filters, out_filters, activation=block_arch.activation))
            in_filters = out_filters
        self.in_filters = in_filters
        return layer_X


class ResidualBlockArch:
    def __init__(self, name, num_layers, filters, kernel_size=3, activation="relu", block=Bottleneck):
        self.name = name
        self.num_layers = num_layers
        self.filters = filters
        self.activation = activation
        self.block = block


resnet_block_arch = [
    ResidualBlockArch("layer1", 3, 64),
    ResidualBlockArch("layer2", 4, 128),
    ResidualBlockArch("layer2", 6, 256),
    ResidualBlockArch("layer2", 3, 512),
]


SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
# Construct a tf.data.Dataset
dataset, info = tfds.load(name="cifar10", as_supervised=True, with_info=True)
ds_train, ds_test = dataset["train"], dataset["test"]

# return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,**kwargs)
model = ResNet(resnet_block_arch)
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.build(input_shape=(None, 32, 32, 3))
model.summary()
for X_batch, y_batch in ds_train.batch(2).take(1):
    X_batch = X_batch / 255
    y_pred = model(X_batch)
print("y_pred {}".format(y_pred))

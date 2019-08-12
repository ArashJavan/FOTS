from tensorflow import keras


class Resnet50Wrapper():
    layers_out = ["activation_9", "activation_21", "activation_39", "activation_48"]

    def __init__(self, input_shape, **kwargs):
        self._bb_model = keras.applications.resnet50.ResNet50(input_shape=input_shape, include_top=False)
        self._bb_model.trainable = False

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
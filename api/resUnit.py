from keras.layers import Activation, Add, BatchNormalization, Conv2D
from tensorflow import keras


class ResUnit(keras.layers.Layer):
    def __init__(self, filters=16, strides=(1, 1), **kwargs):
        super(ResUnit, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides

        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(
            filters, kernel_size=(3, 3), strides=strides, padding="same"
        )
        self.act1 = Activation("relu")

        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")

        self.downsample_conv = Conv2D(filters, kernel_size=(1, 1), strides=strides)
        self.downsample_bn = BatchNormalization()

        self.add = Add()
        self.act2 = Activation("relu")

    def call(self, inputs):
        shortcut = inputs
        if self.strides != (1, 1) or shortcut.shape[-1] != self.filters:
            shortcut = self.downsample_conv(inputs)
            shortcut = self.downsample_bn(shortcut)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.add([x, shortcut])
        x = self.act2(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "strides": self.strides,
            }
        )
        return config

import tensorflow as tf
from delira.utils.decorators import make_deprecated

conv2d = tf.keras.layers.Conv2D
maxpool2d = tf.keras.layers.MaxPool2D
dense = tf.keras.layers.Dense
relu = tf.keras.layers.ReLU
gap2d = tf.keras.layers.GlobalAveragePooling2D
batchnorm2d = tf.keras.layers.BatchNormalization
add = tf.keras.layers.Add


def get_image_format_and_axis():
    """
    helper function to read out keras image_format and convert to axis
    dimension

    Returns
    -------
    str
        image data format (either "channels_first" or "channels_last")
    int
        integer corresponding to the channel_axis (either 1 or -1)
    """
    image_format = tf.keras.backend.image_data_format()
    if image_format == "channels_first":
        return image_format, 1
    elif image_format == "channels_last":
        return image_format, -1
    else:
        raise RuntimeError(
            "Image format unknown, got: {}".format(image_format)
        )


class ResBlock(tf.keras.Model):
    def __init__(self, filters_in: int, filters: int,
                 strides: tuple, kernel_size: int, bias=False):
        super(ResBlock, self).__init__()

        _, _axis = get_image_format_and_axis()

        self.identity = None
        if filters_in != filters:
            self.identity = conv2d(
                filters=filters, strides=strides[0],
                kernel_size=1, padding='same', use_bias=bias)
            self.bnorm_identity = batchnorm2d(axis=_axis)

        self.conv_1 = conv2d(
            filters=filters, strides=strides[0],
            kernel_size=kernel_size,
            padding='same', use_bias=bias)
        self.batchnorm_1 = batchnorm2d(axis=_axis)

        self.conv_2 = conv2d(
            filters=filters, strides=strides[1],
            kernel_size=kernel_size,
            padding='same', use_bias=bias)
        self.batchnorm_2 = batchnorm2d(axis=_axis)

        self.relu = relu()
        self.add = add()

    def call(self, inputs, training=None):

        if self.identity:
            identity = self.identity(inputs)
            identity = self.bnorm_identity(identity, training=training)
        else:
            identity = inputs

        x = self.conv_1(inputs)
        x = self.batchnorm_1(x, training=training)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batchnorm_2(x, training=training)
        x = self.add([x, identity])
        x = self.relu(x)

        return x


class ResNet18(tf.keras.Model):
    @make_deprecated("own repository to be announced")
    def __init__(self, num_classes=None, bias=False):

        super(ResNet18, self).__init__()

        _image_format, _axis = get_image_format_and_axis()

        self.conv1 = conv2d(filters=64, strides=2, kernel_size=7,
                            padding='same', use_bias=bias)
        self.batchnorm1 = batchnorm2d(axis=_axis)
        self.relu = relu()
        self.pool1 = maxpool2d(pool_size=3, strides=2)

        self.block_2_1 = ResBlock(filters_in=64, filters=64,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)

        self.block_2_2 = ResBlock(filters_in=64, filters=64,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)

        self.block_3_1 = ResBlock(filters_in=64, filters=128,
                                  strides=(2, 1), kernel_size=3,
                                  bias=bias)

        self.block_3_2 = ResBlock(filters_in=128, filters=128,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)

        self.block_4_1 = ResBlock(filters_in=128, filters=256,
                                  strides=(2, 1), kernel_size=3,
                                  bias=bias)

        self.block_4_2 = ResBlock(filters_in=256, filters=256,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)

        self.block_5_1 = ResBlock(filters_in=256, filters=512,
                                  strides=(2, 1), kernel_size=3,
                                  bias=bias)

        self.block_5_2 = ResBlock(filters_in=512, filters=512,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)
        self.dense = dense(num_classes)
        self.gap = gap2d(data_format=_image_format)

    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.batchnorm1(x, training=training)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.block_2_1(x, training=training)
        x = self.block_2_2(x, training=training)

        x = self.block_3_1(x, training=training)
        x = self.block_3_2(x, training=training)

        x = self.block_4_1(x, training=training)
        x = self.block_4_2(x, training=training)

        x = self.block_5_1(x, training=training)
        x = self.block_5_2(x, training=training)

        x = self.gap(x)
        x = self.dense(x)

        return x

import tensorflow as tf

conv2d = tf.keras.layers.Conv2D
maxpool2d = tf.keras.layers.MaxPool2D
dense = tf.keras.layers.Dense
relu = tf.keras.layers.ReLU
gap2d = tf.keras.layers.GlobalAveragePooling2D
batchnorm2d = tf.keras.layers.BatchNormalization

class ResNet18(tf.keras.Model):
    def __init__(self, num_classes=None):
        super(ResNet18, self).__init__()
        self.conv1 = conv2d(filters=64, strides=2, kernel_size=7,
                            padding='same')
        self.batchnorm1 = batchnorm2d(axis=1)
        self.pool1 = maxpool2d(pool_size=3, strides=2)

        self.conv2_1 = conv2d(filters=64, strides=1, kernel_size=3,
                              padding='same')
        self.conv2_2 = conv2d(filters=64, strides=1, kernel_size=3,
                              padding='same')

        self.conv3_1 = conv2d(filters=128, strides=2, kernel_size=3,
                              padding='same')
        self.conv3_2 = conv2d(filters=128, strides=1, kernel_size=3,
                              padding='same')

        self.conv4_1 = conv2d(filters=256, strides=2, kernel_size=3,
                              padding='same')
        self.conv4_2 = conv2d(filters=256, strides=1, kernel_size=3,
                              padding='same')

        self.conv5_1 = conv2d(filters=512, strides=2, kernel_size=3,
                              padding='same')
        self.conv5_2 = conv2d(filters=512, strides=1, kernel_size=3,
                              padding='same')

        self.gap = gap2d()
        self.dense1 = dense(1000)
        self.dense2 = dense(num_classes)
        self.relu = relu()

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)

        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)

        x = self.gap(x)
        x = self.dense1(x)

        x = self.dense2(x)
        return x

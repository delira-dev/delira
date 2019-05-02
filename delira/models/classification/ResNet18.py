import tensorflow as tf

conv2d = tf.keras.layers.Conv2D
maxpool2d = tf.keras.layers.MaxPool2D
dense = tf.keras.layers.Dense
relu = tf.keras.layers.ReLU
gap2d = tf.keras.layers.GlobalAveragePooling2D
batchnorm2d = tf.keras.layers.BatchNormalization
add = tf.keras.layers.Add

class ResNet18(tf.keras.Model):
    def __init__(self, num_classes=None):
        super(ResNet18, self).__init__()
        self.conv1 = conv2d(filters=64, strides=2, kernel_size=7,
                            padding='same')
        self.batchnorm1 = batchnorm2d(axis=1)
        self.pool1 = maxpool2d(pool_size=3, strides=2)

        self.conv2_1 = conv2d(filters=64, strides=1, kernel_size=3,
                              padding='same')
        self.batchnorm2_1 = batchnorm2d(axis=1)
        self.conv2_2 = conv2d(filters=64, strides=1, kernel_size=3,
                              padding='same')
        self.batchnorm2_2 = batchnorm2d(axis=1)

        self.identity_3 = conv2d(filters=128, strides=2, kernel_size=1,
                              padding='same')                              
        self.batchnorm3 = batchnorm2d(axis=1)
        self.conv3_1 = conv2d(filters=128, strides=2, kernel_size=3,
                              padding='same')
        self.batchnorm3_1 = batchnorm2d(axis=1)
        self.conv3_2 = conv2d(filters=128, strides=1, kernel_size=3,
                              padding='same')
        self.batchnorm3_2 = batchnorm2d(axis=1)

        self.identity_4 = conv2d(filters=256, strides=2, kernel_size=1,
                              padding='same')                              
        self.batchnorm4 = batchnorm2d(axis=1)
        self.conv4_1 = conv2d(filters=256, strides=2, kernel_size=3,
                              padding='same')
        self.batchnorm4_1 = batchnorm2d(axis=1)
        self.conv4_2 = conv2d(filters=256, strides=1, kernel_size=3,
                              padding='same')
        self.batchnorm4_2 = batchnorm2d(axis=1)

        self.identity_5 = conv2d(filters=512, strides=2, kernel_size=1,
                              padding='same')                              
        self.batchnorm5 = batchnorm2d(axis=1)
        self.conv5_1 = conv2d(filters=512, strides=2, kernel_size=3,
                              padding='same')
        self.batchnorm5_1 = batchnorm2d(axis=1)
        self.conv5_2 = conv2d(filters=512, strides=1, kernel_size=3,
                              padding='same')
        self.batchnorm5_2 = batchnorm2d(axis=1)

        self.dense1 = dense(1000)
        self.dense2 = dense(num_classes)
       
        self.gap = gap2d()
        self.relu = relu()
        self.add = add()

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.pool1(x)

        id = x
        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.batchnorm2_2(x)
        x = self.add([x, id])
        x = self.relu(x)

        id = self.identity_3(x)
        id = self.batchnorm3(id)
        x = self.conv3_1(x)
        x = self.batchnorm3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.batchnorm3_2(x)
        x = self.add([x, id])
        x = self.relu(x)

        id = self.identity_4(x)
        id = self.batchnorm4(id)
        x = self.conv4_1(x)
        x = self.batchnorm4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.batchnorm4_2(x)
        x = self.add([x, id])
        x = self.relu(x)

        id = self.identity_5(x)
        id = self.batchnorm5(id)
        x = self.conv5_1(x)
        x = self.batchnorm5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.batchnorm5_2(x)
        x = self.add([x, id])
        x = self.relu(x)

        x = self.gap(x)
        x = self.dense1(x)

        x = self.dense2(x)
        return x

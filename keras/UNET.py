import tensorflow as tf

class ConvBlock(tf.keras.Layer):
    """See keras example
    https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#layers_are_recursively_composable
    Convolutional block with a Conv2D layer followed by BatchNormalization and ReLU activation.
    """
    def __init__(self, filters, kernel_size=3, padding='valid', batchnorm=True, trainable=True, **kwargs):
        super().__init__(**kwargs)
        # This call will create the weights
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size,  padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        if not trainable:
            self.conv.trainable = False
            self.bn.trainable = False

        self.activation = tf.keras.layers.Activation('relu')
        self.batchnorm = batchnorm

    def call(self, inputs, activation=True):
        x = self.conv(inputs)
        if self.batchnorm:
            x = self.bn(x)
        return self.activation(x) if activation else x

class SideConvBlock(tf.keras.Layer):
    """Convolutional block with two Conv2D layers, optional BatchNormalization and ReLU activation.
    This is a side block in the U-Net architecture.
    """
    def __init__(self, filters, kernel_size=3, padding='valid', batchnorm=True, trainable=True, **kwargs):
        super(SideConvBlock, self).__init__(**kwargs)
        self.conv1 = ConvBlock(filters, kernel_size, padding, batchnorm, trainable)
        self.conv2 = ConvBlock(filters, kernel_size, padding, batchnorm, trainable)

    def call(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(x)

class DownSampleBlock(tf.keras.Layer):
    """Downsampling block with a Conv2D layer followed by BatchNormalization and ReLU activation, and MaxPooling."""
    def __init__(self, filters, kernel_size=3, padding='valid', batchnorm=True, trainable=True, **kwargs):
        super(DownSampleBlock, self).__init__(**kwargs)
        self.conv = SideConvBlock(filters, kernel_size, padding, batchnorm, trainable)
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x = self.conv(inputs)
        return self.pool(x), x

class EncoderBlock(tf.keras.Layer):
    """Encoder block that consists of multiple DownSampleBlock layers.
    This is used to create the encoder part of the U-Net architecture.
    """
    def __init__(self, input_shape, num_filters=None, batchnorm=False, padding='valid', trainable=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.input_layer = tf.keras.Input(shape=input_shape)
        self.num_filters = num_filters if num_filters is not None else [64, 128, 256, 512]
        self.batchnorm = batchnorm
        self.padding = padding
        self.downsample_blocks = [DownSampleBlock(filters, padding=padding, batchnorm=batchnorm, trainable=trainable,
                                                  **kwargs) for filters in self.num_filters]

    def call(self, inputs):
        skips = []
        x = inputs
        for downsample_block in self.downsample_blocks:
            x, skip = downsample_block(x)
            skips.append(skip)
        return x, skips

class BottleneckBlock(tf.keras.Layer):
    """Bottleneck block that consists of a SideConvBlock.
    This is used to create the bottleneck part of the U-Net architecture.
    """
    def __init__(self, filters, kernel_size=3, padding='valid', batchnorm=True, trainable=True, **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        self.conv = SideConvBlock(filters, kernel_size, padding, batchnorm, trainable)

    def call(self, inputs):
        return self.conv(inputs)

class UpConvBlock(tf.keras.Layer):
    def __init__(self, num_filters, padding='valid', batchnorm=True, trainable=True, **kwargs):
        super(UpConvBlock, self).__init__(**kwargs)  # Ensure the base class is initialized
        self.up = tf.keras.layers.UpSampling2D((2, 2))
        self.conv = ConvBlock(num_filters, padding=padding, batchnorm=batchnorm, trainable=trainable, **kwargs)

    def call(self, inputs):
        x = self.up(inputs)
        return self.conv(x, activation=False)

class UpSampleBlock(tf.keras.Layer):
    def __init__(self, num_filters=256, padding='valid', batchnorm=True, trainable=True, **kwargs):
        super(UpSampleBlock, self).__init__()
        self.up_conv = UpConvBlock(num_filters, padding=padding, batchnorm=batchnorm, trainable=trainable, **kwargs)
        self.side_conv = SideConvBlock(num_filters, padding=padding, batchnorm=batchnorm, trainable=trainable, **kwargs)

    def call(self, inputs, skip_connection):
        x = self.up_conv(inputs)
        # Resize the skip connection to match the size of x
        #skip_connection = tf.image.resize(skip_connection, size=x.shape[1:3])

        # Paper says to crop the skip connection to match the size of x
        #We asume that channel dimension is first. (B, H, W, C)
        H, W = x.shape[1:3]
        assert skip_connection.shape[1] >= H and skip_connection.shape[2] >= W
        skip_connection = tf.keras.layers.CenterCrop(H, W, data_format='channels_last')(skip_connection)
        x = tf.keras.layers.Concatenate()([x, skip_connection])

        return self.side_conv(x)

class DecoderBlock(tf.keras.Layer):
    def __init__(self, num_filters=None, padding='valid', batchnorm=True, trainable=True, **kwargs):
        super(DecoderBlock, self).__init__()
        self.up_sample_blocks = []
        if num_filters is None:
            num_filters = [512, 256, 128, 64]

        for filters in num_filters:
            self.up_sample_blocks.append(UpSampleBlock(filters, padding=padding, batchnorm=batchnorm,
                                                       trainable=trainable, **kwargs))

    def call(self, inputs, skips):
        x = inputs
        for i, up_sample_block in enumerate(self.up_sample_blocks):
            x = up_sample_block(x, skips[-(i + 1)])

        return x


class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, num_classes=1, trainable=True, padding='valid', batchnorm=True, **kwargs):
        super(OutputBlock, self).__init__(**kwargs)
        self.conv = ConvBlock(num_classes, kernel_size=1, padding=padding, batchnorm=batchnorm, trainable=trainable)
        self.num_classes = num_classes

    def call(self, inputs):
        x = self.conv(inputs, activation=False)
        if self.num_classes > 1:
            # (B, C, H, W)
            x = tf.nn.softmax(x, axis=1)
        else:
            # (B, H, W, C)
            x = tf.nn.sigmoid(x)

        return x

def test_output_block():
    # (B, C, H, W) input shape
    # Define the dimensions
    B, C, H, W = 1, 2, 2, 2  # Example dimensions

    # Generate the random tensor
    input_tensor = tf.random.uniform(shape=(B, C, H, W), minval=0, maxval=1)
    # Apply softmax along the channel dimension
    print("Input Tensor:")
    print(input_tensor)
    output_block = OutputBlock(num_classes=2, trainable=True, padding='valid', batchnorm=True)
    output = output_block(input_tensor)
    print("Output Tensor:")
    print(output)

    print("Output shape:", output.shape)

class UNET(tf.keras.Model):
    def __init__(self, encoder_trainable=True, num_filters=None, padding='valid', batchnorm=True, classes=1, **kwargs):
        super(UNET, self).__init__( **kwargs)
        self.encoder = EncoderBlock(input_shape=(512, 512, 3), num_filters=num_filters,
                                    batchnorm=batchnorm, padding=padding, trainable=encoder_trainable)

        self.bottleneck = BottleneckBlock(filters=1024, kernel_size=3, padding=padding, batchnorm=batchnorm)

        self.decoder = DecoderBlock(num_filters=num_filters, padding=padding, batchnorm=batchnorm)

        self.output_block = OutputBlock(num_classes=classes, padding=padding, batchnorm=batchnorm)

    def call(self, inputs):
        x, skips = self.encoder(inputs)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.output_block(x)

    def build(self, input_shape):
        super(UNET, self).build(input_shape)
        # Call the model once to initialize all layers
        self.call(tf.keras.Input(shape=input_shape[1:]))




if __name__ == "__main__":
    model = UNET(encoder_trainable=True, padding='same', batchnorm=False)
    model.build((None, 572, 572, 3))  # Input shape (batch_size, channels, height, width)
    tf.keras.utils.plot_model(model, "my_first_model.png")
    print(model.summary())

"""MobileNet v3 models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

class MobileNetBase:
    def __init__(self, input_tensor, n_class):
        self.input = input_tensor
        self.n_class = n_class

    def _relu6(self, x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.

        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """
        if nl == 'HS':
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':
            x = Activation(self._relu6)(x)

        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """Convolution Block
        This function defines a 2D convolution operation with BN and activation.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)
        tchannel = input_shape[channel_axis] * e
        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if squeeze:
            x = Lambda(lambda x: x * self._squeeze(x))(x)

        x = self._return_activation(x, nl)

        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass

"""MobileNet v3 Large models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""



class MobileNetV3_Large(MobileNetBase):
    def __init__(self, input_tensor, n_class):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.

        # Returns
            MobileNetv2 model.
        """
        super(MobileNetV3_Large, self).__init__(input_tensor, n_class)

    def build(self, plot=False):
        """build MobileNetV3 Large.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = self.input

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')
        x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)

        output = Reshape((self.n_class,))(x)

        model = Model(inputs, output)

        if plot:
            plot_model(model, to_file='images/MobileNetv3_large.png', show_shapes=True)

        return model


class MobileNetV3_Small(MobileNetBase):
    def __init__(self, input_tensor, n_class):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.

        # Returns
            MobileNetv2 model.
        """
        super(MobileNetV3_Small, self).__init__(input_tensor, n_class)


    def build(self, plot=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = self.input

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')
        x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)

        output = Reshape((self.n_class,))(x)

        model = Model(inputs, output)

        if plot:
            plot_model(model, to_file='images/MobileNetv3_small.png', show_shapes=True)

        return model

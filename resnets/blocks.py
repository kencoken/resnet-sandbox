import logging

from keras.layers import (
    Activation
)
from keras.layers.convolutional import (
    Convolution2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization

log = logging.getLogger(__name__)


def bn_relu(name, num):

    def f(input):
        bn = BatchNormalization(mode=0, axis=1, name='{}_bn{}'.format(name, num))(input)
        relu = Activation('relu', name='{}_relu{}'.format(name, num))(bn)
        return relu

    return f

def residual(name, stride, output_sz):

    def f(input):
        conv1 = Convolution2D(nb_filter=output_sz, nb_row=3, nb_col=3,
                              subsample=(stride, stride), border_mode='same',
                              init='he_normal',
                              name='{}_conv1'.format(name))(input)

        bn_relu1 = bn_relu(name, 1)(conv1)
        conv2 = Convolution2D(nb_filter=output_sz, nb_row=3, nb_col=3,
                              subsample=(1, 1), border_mode='same',
                              init='he_normal',
                              name='{}_conv2'.format(name))(bn_relu1)

        return conv2

    return f

def residual_bottleneck(name, stride, output_sz):

    def f(input):
        bottleneck_sz = int(output_sz / 4)

        conv1 = Convolution2D(nb_filter=bottleneck_sz, nb_row=1, nb_col=1,
                              subsample=(stride, stride), border_mode='same',
                              init='he_normal',
                              name='{}_conv1'.format(name))(input)

        bn_relu1 = bn_relu(name, 1)(conv1)
        conv2 = Convolution2D(nb_filter=bottleneck_sz, nb_row=3, nb_col=3,
                              subsample=(1, 1), border_mode='same',
                              init='he_normal',
                              name='{}_conv2'.format(name))(bn_relu1)

        bn_relu2 = bn_relu(name, 2)(conv2)
        conv3 = Convolution2D(nb_filter=output_sz, nb_row=1, nb_col=1,
                              subsample=(1, 1), border_mode='same',
                              init='he_normal',
                              name='{}_conv3'.format(name))(bn_relu2)

        return conv3

    return f

def shortcut(name, num, stride):

    def f(input):
        if stride == 1:
            return input
        else:
            return AveragePooling2D(pool_size=(stride, stride), border_mode='same',
                                    name='{}_pool{}'.format(name, num))(input)

    return f

def shortcut_upsample(name, num, stride, output_sz):

    def f(input):
        conv = Convolution2D(nb_filter=output_sz, nb_row=1, nb_col=1,
                             subsample=(stride, stride), border_mode='same',
                             init='he_normal',
                             name='{}_conv{}'.format(name, num))(input)
        return conv

    return f

import logging

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization

import tensorflow as tf

log = logging.getLogger('')


def bn_relu(name, num):

    def f(input):
        bn = BatchNormalization(mode=0, axis=1, name='{}_bn{}'.format(name, num))(input)
        relu = Activation('relu', name='{}_relu{}'.format(name, num))(bn)
        return relu

    return f


def basic_block(output_sz, stride, name, preactivate=True, upsample=False):

    if upsample:
        log.warning('Upsample has no effect for blocks of type basic_block')

    def f(input):

        with tf.name_scope(name):

            if preactivate:
                input = bn_relu(name, 0)(input)
            conv1 = Convolution2D(nb_filter=output_sz, nb_row=3, nb_col=3,
                                  subsample=(stride, stride), border_mode='same',
                                  init='he_normal',
                                  name='{}_conv1'.format(name))(input)

            bn_relu1 = bn_relu(name, 1)(conv1)
            conv2 = Convolution2D(nb_filter=output_sz, nb_row=3, nb_col=3,
                                  subsample=(stride, stride), border_mode='same',
                                  init='he_normal',
                                  name='{}_conv2'.format(name))(bn_relu1)

            residual = conv2
            shortcut = input

            return merge([shortcut, residual], mode='sum')

    return f

def bottleneck_block(name, output_sz, stride=1, preactivate=True, upsample=False):

    # block sub-components ---

    def main_path_(input):

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

    def regular_shortcut_path_(input):

        if stride == 1:
            return input
        else:
            return AveragePooling2D(pool_size=(stride, stride), border_mode='same')(input)

    def upsample_shortcut_path_(input):

        conv4 = Convolution2D(nb_filter=output_sz, nb_row=1, nb_col=1,
                              subsample=(stride, stride), border_mode='same',
                              init='he_normal',
                              name='{}_conv4'.format(name))(input)

        return conv4

    # regular / upsample blocks ---

    def regular_block(input):

        if preactivate:
            bn_relu0 = bn_relu(name, 0)(input)
            residual = main_path_(bn_relu0)
        else:
            residual = main_path_(input)

        shortcut = regular_shortcut_path_(input)

        return merge([shortcut, residual], mode='sum')

    def upsample_block(input):

        # in upsample block, first bn/relu is shared across both branches
        if preactivate:
            input = bn_relu(name, 0)(input)

        residual = main_path_(input)
        shortcut = upsample_shortcut_path_(input)

        return merge([shortcut, residual], mode='sum')

    # block factory ---

    def f(input):

        with tf.name_scope(name):

            if not upsample:
                return regular_block(input)
            else:
                return upsample_block(input)

    return f

def build_block_group(name, block_fn, n, nb_filter, stride, first_type=None):

    assert(first_type in [None, 'upsample', 'first'])

    def f(input):
        print(name)
        print(stride)

        res = block_fn(name='{}_1'.format(name),
                       output_sz=nb_filter, stride=stride,
                       preactivate=(first_type!='first'),
                       upsample=(first_type=='upsample' or first_type=='first'))(input)

        for i in range(1, n):
            subblock_stride = 1  # only adjust stride for first block in group
            print(subblock_stride)
            # if name == 'res3':
            #     print(i)
            #     merge([res, input], mode='sum')
                
            res = block_fn(name='{}_{}'.format(name, i+1),
                           output_sz=nb_filter, stride=subblock_stride)(res)

        return res

    return f

def resnet():
    
    input = Input(shape=(224, 224, 3))

    with tf.name_scope('conv1'):
        conv1 = Convolution2D(nb_filter=64, nb_row=7, nb_col=7,
                              subsample=(2, 2), border_mode='same',
                              init='he_normal', name='conv1')(input)
        conv1_bn = BatchNormalization(mode=0, axis=1)(conv1)
        conv1_relu = Activation('relu', name='relu1')(conv1_bn)

    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same',
                         name='pool1')(conv1_relu)

    res2 = build_block_group(name='res2',
                             block_fn=bottleneck_block, n=3,
                             nb_filter=256, stride=1,
                             first_type='first')(pool1)

    res3 = build_block_group(name='res3',
                             block_fn=bottleneck_block, n=8,
                             nb_filter=512, stride=2,
                             first_type='upsample')(res2)

    res4 = build_block_group(name='res4',
                             block_fn=bottleneck_block, n=36,
                             nb_filter=1024, stride=2,
                             first_type='upsample')(res3)

    res5 = build_block_group(name='res5',
                             block_fn=bottleneck_block, n=3,
                             nb_filter=2048, stride=2,
                             first_type='upsample')(res4)

    res5_bn_relu = bn_relu('final', 1)(res5)

    pool2 = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode='same')(res5_bn_relu)
    fc1 = Flatten()(pool2)
    fc2 = Dense(output_dim=1000, init='he_normal', activation='softmax')(fc1)

    model = Model(input=input, output=fc2)

    return model

def main():
    import time
    start = time.time()
    model = resnet()
    duration = time.time() - start
    print('{} s to make model'.format(duration))

    start = time.time()
    model.output
    duration = time.time() - start
    print('{} s to get output'.format(duration))

    model.summary()

    # start = time.time()
    # model.compile(loss='categorical_crossentropy', optimizer='sgd')
    # duration = time.time() - start
    # print('{} s to compile'.format(duration))

if __name__ == '__main__':
    main()
    

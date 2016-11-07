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
from keras import backend as K

log = logging.getLogger('')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def bn_relu(name, num):

    def f(input):
        bn = BatchNormalization(mode=0, axis=1, name='{}_bn{}'.format(name, num))(input)
        relu = Activation('relu', name='{}_relu{}'.format(name, num))(bn)
        return relu

    return f


def basic_block(output_sz, stride, name, preactivate=True, upsample=False):

    def regular_shortcut_path_(input):

        if stride == 1:
            return input
        else:
            return AveragePooling2D(pool_size=(stride, stride), border_mode='same')(input)

    def upsample_shortcut_path_(input):

        conv = Convolution2D(nb_filter=output_sz, nb_row=1, nb_col=1,
                             subsample=(stride, stride), border_mode='same',
                             init='he_normal',
                             name='{}_conv4'.format(name))(input)

        return conv

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
                                  subsample=(1, 1), border_mode='same',
                                  init='he_normal',
                                  name='{}_conv2'.format(name))(bn_relu1)

            residual = conv2
            shortcut = upsample_shortcut_path_(input) if upsample else regular_shortcut_path_(input)

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

        conv = Convolution2D(nb_filter=output_sz, nb_row=1, nb_col=1,
                             subsample=(stride, stride), border_mode='same',
                             init='he_normal',
                             name='{}_conv4'.format(name))(input)

        return conv

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

def resnet(dataset='imagenet', layer_count=None):

    def imagenet(layer_count=152):

        cfgs = {
            18: {'groups': [(64, 2), (128, 2), (256, 2), (512, 2)], 'block_fn': basic_block},
            34: {'groups': [(64, 3), (128, 4), (256, 6), (512, 3)], 'block_fn': basic_block},
            50: {'groups': [(256, 3), (512, 4), (1024, 6), (2048, 3)], 'block_fn': bottleneck_block},
            101: {'groups': [(256, 3), (512, 4), (1024, 23), (2048, 3)], 'block_fn': bottleneck_block},
            152: {'groups': [(256, 3), (512, 8), (1024, 36), (2048, 3)], 'block_fn': bottleneck_block}
        }
        assert layer_count in cfgs, 'layer_count should be one of: {}'.format(cfgs.keys())
        group_lists = list(zip(*cfgs[layer_count]['groups']))
        cfg = {'filter_sz': group_lists[0], 'group_sz': group_lists[1], 'block_fn': cfgs[layer_count]['block_fn']}

        # start of model definition ---
    
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
                                 block_fn=cfg['block_fn'], n=cfg['group_sz'][0],
                                 nb_filter=cfg['filter_sz'][0], stride=1,
                                 first_type='first')(pool1)

        res3 = build_block_group(name='res3',
                                 block_fn=cfg['block_fn'], n=cfg['group_sz'][1],
                                 nb_filter=cfg['filter_sz'][1], stride=2,
                                 first_type='upsample')(res2)

        res4 = build_block_group(name='res4',
                                 block_fn=cfg['block_fn'], n=cfg['group_sz'][2],
                                 nb_filter=cfg['filter_sz'][2], stride=2,
                                 first_type='upsample')(res3)

        res5 = build_block_group(name='res5',
                                 block_fn=cfg['block_fn'], n=cfg['group_sz'][3],
                                 nb_filter=cfg['filter_sz'][3], stride=2,
                                 first_type='upsample')(res4)

        res5_bn_relu = bn_relu('final', 1)(res5)

        pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode='valid')(res5_bn_relu)
        fc1 = Flatten()(pool2)
        fc2 = Dense(output_dim=1000, init='he_normal', activation='softmax')(fc1)

        model = Model(input=input, output=fc2)

        return model

    def cifar10(layer_count=32):

        valid_layer_counts = [20, 32, 44, 56, 110, 1202]
        assert layer_count in valid_layer_counts, 'layer_count should be one of: {}'.format(valid_layer_counts)

        # start of model definition ---

        input = Input(shape=(32, 32, 3))

        conv1 = Convolution2D(nb_filter=16, nb_row=3, nb_col=3,
                              subsample=(1, 1), border_mode='same',
                              init='he_normal', name='conv1')(input)

        res2 = build_block_group(name='res2',
                                 block_fn=basic_block, n=layer_count,
                                 nb_filter=16, stride=1)(conv1)

        res3 = build_block_group(name='res3',
                                 block_fn=basic_block, n=layer_count,
                                 nb_filter=32, stride=2, first_type='upsample')(res2)

        res4 = build_block_group(name='res4',
                                 block_fn=basic_block, n=layer_count,
                                 nb_filter=64, stride=2, first_type='upsample')(res3)

        res4_bn_relu = bn_relu('final', 1)(res4)

        pool1 = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode='valid')(res4_bn_relu)
        fc1 = Flatten()(pool1)
        fc2 = Dense(output_dim=10, init='he_normal', activation='softmax')(fc1)

        model = Model(input=input, output=fc2)

        return model

    # map and create model ---

    dataset_fn_map = dict(
        imagenet=imagenet,
        cifar10=cifar10
    )
    assert dataset in dataset_fn_map, 'dataset should be one of: {}'.format(dataset_fn_map.keys())
    dataset_fn = dataset_fn_map[dataset]

    return dataset_fn() if layer_count is None else dataset_fn(layer_count)

def main():
    # dataset = 'imagenet'
    dataset = 'cifar10'
    
    import time
    start = time.time()
    model = resnet(dataset)
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
    

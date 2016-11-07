import logging

from keras.models import Model
from keras.layers import (
    Input,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)

import tensorflow as tf

from . import blocks

log = logging.getLogger(__name__)


def basic_block(output_sz, stride, name, preactivate=True, upsample=False):

    def main_path_(input):
        return blocks.residual(name, stride, output_sz)(input)

    def shortcut_path_(input):
        if upsample:
            return blocks.shortcut_upsample(name, 3, stride, output_sz)(input)
        else:
            return blocks.shortcut(name, 3, stride)(input)

    def f(input):
        with tf.name_scope(name):
            if preactivate:
                bn_relu0 = blocks.bn_relu(name, 0)(input)

            residual = main_path_(bn_relu0)
            shortcut = shortcut_path_(input)

            return merge([shortcut, residual], mode='sum')

    return f

def bottleneck_block(name, output_sz, stride=1, preactivate=True, upsample=False):

    def main_path_(input):
        return blocks.residual_bottleneck(name, stride, output_sz)(input)

    def shortcut_path_(input):
        if upsample:
            return blocks.shortcut_upsample(name, 4, stride, output_sz)(input)
        else:
            return blocks.shortcut(name, 4, stride)(input)

    def f(input):
        with tf.name_scope(name):
            if preactivate:
                bn_relu0 = blocks.bn_relu(name, 0)(input)

                residual = main_path_(bn_relu0)
                # in upsample block, first bn/relu is shared across both branches
                shortcut = shortcut_path_(bn_relu0) if upsample else shortcut_path_(input)

            else:
                residual = main_path_(input)
                shortcut = shortcut_path_(input)

            return merge([shortcut, residual], mode='sum')

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

def build_model(dataset='imagenet', layer_count=None):

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
            conv1_relu = blocks.bn_relu('initial', 1)(conv1)

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

        res5_bn_relu = blocks.bn_relu('final', 1)(res5)

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

        res4_bn_relu = blocks.bn_relu('final', 1)(res4)

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

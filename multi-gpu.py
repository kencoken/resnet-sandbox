import os, shutil
import time
import math
from contextlib import contextmanager
import logging

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ProgbarLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda

from resnets.callbacks import TensorBoardExtra
from resnets import resnet



def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice no. [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def to_multi_gpu(model, n_gpus=2):
    """Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.
    """
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape,
                             arguments={'n_gpus': n_gpus, 'part': g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)

    return Model(input=[x], output=merged)

@contextmanager
def time_block(description):
    start = time.time()
    yield
    duration = time.time() - start
    print('{:.4f} s to {}'.format(duration, description))

## ---

def prepare_model(dataset, layer_count=None):

    with time_block('make model'):
        model = resnet.build_model(dataset, layer_count)

    return model

def train_model(model):

    with time_block('load cifar10'):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # split into 47k train, 3k val //45k train, 5k val
        val_sz = 5000; train_sz = X_train.shape[0] - val_sz
        val_sz_limit = 2500
        val_idxs = np.random.choice(X_train.shape[0], size=val_sz, replace=False)
        X_val = X_train[val_idxs]
        Y_val = Y_train[val_idxs]
        mask = np.ones(X_train.shape[0], np.bool)
        mask[val_idxs] = 0
        X_train = X_train[mask]
        Y_train = Y_train[mask]

    datagen_train = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        fill_mode='constant', cval=0.,
        horizontal_flip=True
    )
    datagen_test = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    with time_block('fit data mean'):
        datagen_train.fit(X_train)
        datagen_test.mean = datagen_train.mean
        datagen_test.std = datagen_train.std

    print('Start training...')
    batch_sz = 128
    base_lr = 0.1
    to_train_epochs = lambda x: int(x*128/train_sz)
    nb_epoch = to_train_epochs(64000)  # equivalent to 64k iters in paper
    epoch_divider = 10

    prog_callback = ProgbarLogger()
    
    hist = model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=batch_sz),
                               validation_data=datagen_test.flow(X_val, Y_val, batch_size=batch_sz*4),
                               nb_val_samples=val_sz,
                               samples_per_epoch=int(len(X_train)/epoch_divider), nb_epoch=nb_epoch*epoch_divider,
                               callbacks=[prog_callback])

# To use just take any model and set model = to_multi_gpu(model).
# model.fit() and model.predict() should work without any change.
#
# Example:

def main():
    model = resnet.build_model('cifar10', 32)
    model = to_multi_gpu(model, n_gpus=2)

    with time_block('compile'):
        sgd = SGD(lr=0.1, decay=0.0001, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['categorical_accuracy'])

    model.summary()
    
    train_model(model)

if __name__ == '__main__':
    main()

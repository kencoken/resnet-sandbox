import os, shutil
import time
import math
from contextlib import contextmanager
import logging

import numpy as np
import tensorflow as tf

from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ProgbarLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger
from tensorflow.python.client import timeline

from resnets import resnet
from keras_ext.callbacks import TensorBoardExtra
from keras_ext.multi_gpu import to_multi_gpu
from keras_ext.callbacks import ResumableModelCheckpoint
from keras_ext import profiling


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

def train_model(model, batch_sz, debug_epoch=False):

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
    base_lr = 0.1
    to_train_epochs = lambda x: int(x*128/train_sz)
    nb_epoch = to_train_epochs(64000)  # equivalent to 64k iters in paper
    epoch_divider = 10

    if not debug_epoch:
        samples_per_epoch = int(len(X_train)/epoch_divider)
        nb_epoch_fit = nb_epoch*epoch_divider
    else:
        samples_per_epoch = 1000
        nb_epoch_fit = 1

    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists('tmp/tb'):
        os.makedirs('tmp/tb')

    prog_callback = ProgbarLogger()
    tb_callback = TensorBoardExtra(log_dir='tmp/tb/', histogram_freq=1, write_graph=True)
    csv_callback = CSVLogger('tmp/training_log.csv', append=True)
    checkpoint_callback = ResumableModelCheckpoint('tmp/weights.{epoch:04d}-{val_loss:.2f}.h5',
                                                   save_weights_only=True, prune_freq=epoch_divider)
    initial_epoch, chkpt_fname = checkpoint_callback.get_last_checkpoint()

    if initial_epoch > 0:
        print('Resuming training from epoch: {}\n    (loading checkpoint: {})'.format(initial_epoch, chkpt_fname))
        model.load_weights(chkpt_fname)
    
    hist = model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=batch_sz),
                               validation_data=datagen_test.flow(X_val, Y_val, batch_size=batch_sz*4),
                               nb_val_samples=val_sz,
                               samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch_fit,
                               callbacks=[prog_callback, tb_callback, csv_callback, checkpoint_callback],
                               initial_epoch=initial_epoch)

# To use just take any model and set model = to_multi_gpu(model).
# model.fit() and model.predict() should work without any change.
#
# Example:

def main():
    model = resnet.build_model('cifar10', 32)
    model = to_multi_gpu(model, n_gpus=4)
    batch_sz = 128
    generate_timeline = False

    with time_block('compile'):
        sgd = SGD(lr=0.1, decay=0.0001, momentum=0.9, nesterov=True)
        execute_kwargs = profiling.get_profiling_execute_kwargs(enabled=generate_timeline)
        run_metadata = execute_kwargs['run_metadata']
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['categorical_accuracy'],
                      execute_kwargs=execute_kwargs)

    model.summary()
    
    train_model(model, batch_sz=batch_sz, debug_epoch=generate_timeline)

    if generate_timeline:
        profiling.save_timeline('timeline.json', run_metadata)

if __name__ == '__main__':
    main()

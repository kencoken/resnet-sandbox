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

from resnets.callbacks import TensorBoardExtra
from resnets import resnet


log = logging.getLogger('')

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

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

    with time_block('get output'):
        model.output

    with time_block('compile'):
        sgd = SGD(lr=0.1, decay=0.0001, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['categorical_accuracy'])

    model.summary()

    return model

def prepare_data_dirs_():

    dirs = dict(
        log_dir='./data/logs',
        data_dir='./data'
    )
    for fname in os.listdir(dirs['data_dir']):
        fname = os.path.join(dirs['data_dir'], fname)
        os.remove(fname) if os.path.isfile(fname) else shutil.rmtree(fname)
    # for dirname, dir in dirs.items():
    #     if os.path.exists(dir):
    #         shutil.rmtree(dir)
    #     os.makedirs(dir)

    return dirs

def train_model(model):

    dirs = prepare_data_dirs_()

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
        np.save(os.path.join(dirs['data_dir'], 'val_idxs.npy'), val_idxs)
        X_val = X_train[val_idxs]
        Y_val = Y_train[val_idxs]
        mask = np.ones(X_train.shape[0], np.bool)
        mask[val_idxs] = 0
        X_train = X_train[mask]
        Y_train = Y_train[mask]

    datagen_train = ImageDataGenerator(
        featurewise_center=True,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # horizontal_flip=True
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
        # save mean and std
        np.save(os.path.join(dirs['data_dir'], 'mean.npy'), datagen_train.mean)
        np.save(os.path.join(dirs['data_dir'], 'std.npy'), datagen_train.std)

    print('Start training...')
    batch_sz = 32
    base_lr = 0.1
    to_train_epochs = lambda x: int(x*128/train_sz)
    nb_epoch = to_train_epochs(64000)  # equivalent to 64k iters in paper
    epoch_divider = 10

    def lr_schedule(epoch):
        print('epoch is: {}'.format(epoch))
        # equivalent to 32k, 48k iters in paper
        bps = [to_train_epochs(x)*epoch_divider for x in [32000, 48000]]
        n = sum([int(epoch > bp) for bp in bps])
        return base_lr / math.pow(10, n)

    prog_callback = ProgbarLogger()
    tb_callback = TensorBoardExtra(log_dir=dirs['log_dir'], histogram_freq=1, write_graph=True)
    chpt_callback = ModelCheckpoint(os.path.join(dirs['data_dir'],
                                                 'weights.{epoch}-{val_loss:.2f}.h5'),
                                    save_weights_only=True)
    lr_callback = LearningRateScheduler(lr_schedule)

    # flatten validation_data, as currently TensorBoard requires full data rather than iterator
    validation_data = datagen_test.flow(X_val, Y_val, batch_size=val_sz_limit).next()
    
    hist = model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=batch_sz),
                               validation_data=validation_data,
                               samples_per_epoch=int(len(X_train)/epoch_divider), nb_epoch=nb_epoch*epoch_divider,
                               callbacks=[prog_callback, tb_callback, chpt_callback, lr_callback])
    print(hist)

    eval_res = model.evaluate_generator(datagen_test.flow(X_test, Y_test, batch_size=batch_sz), val_samples=X_test.shape[0])
    print(eval_res)

def main():
    # dataset = 'imagenet'; layer_count = 18
    dataset = 'cifar10'; layer_count = 32

    model = prepare_model(dataset, layer_count)
    train_model(model)

if __name__ == '__main__':
    main()
    

import os, shutil
import time
import math
from contextlib import contextmanager
import logging

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ProgbarLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

from resnets import resnet
from resnets.callbacks import TensorBoardExtra
from data_loader.producers import CIFAR10Producer, ImageNetProducer

IMAGE_NET_DIR = '/data/_datasets/ILSVRC2012'

import sys
sys.setrecursionlimit(10000)


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

def prepare_data_dirs_(dataset, experiment_name, clear_root=False):

    dirs = dict(
        root='./data',
        logs=os.path.join(dataset, 'logs', experiment_name),
        snapshots=os.path.join(dataset, 'snapshots', experiment_name)
    )

    # clear old root directory
    if clear_root:
        for fname in os.listdir(dirs['root']):
            fname = os.path.join(dirs['root'], fname)
            os.remove(fname) if os.path.isfile(fname) else shutil.rmtree(fname)

    # ensure all directories exist
    for name, dir in dirs.items():
        if name != 'root':
            full_dir = os.path.join(dirs['root'], dir)
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)

    def get_dir(name):
        return dirs['root'] if name == 'root' else os.path.join(dirs['root'], dirs[name])

    return get_dir

def train_model(model, dataset, experiment_name):

    data_dir = prepare_data_dirs_(dataset, experiment_name)

    with time_block('load dataset'):
        if dataset == 'cifar10':
            producer = CIFAR10Producer(data_dir)
        elif dataset == 'imagenet':
            imnet_dir = IMAGE_NET_DIR
            target_size = (224, 224)
            producer = ImageNetProducer(imnet_dir, target_size=target_size)
        else:
            raise RuntimeError('Unknown dataset!')

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
        datagen_train.fit(producer.train_sample())
        datagen_test.mean = datagen_train.mean
        datagen_test.std = datagen_train.std
        # save mean and std
        np.save(os.path.join(data_dir('snapshots'), 'mean.npy'), datagen_train.mean)
        np.save(os.path.join(data_dir('snapshots'), 'std.npy'), datagen_train.std)

    print('Start training...')
    batch_sz = 32
    base_lr = 0.1
    epoch_divider = 10

    prog_callback = ProgbarLogger()
    tb_callback = TensorBoardExtra(log_dir=data_dir('logs'), histogram_freq=1, write_graph=True)
    chpt_callback = ModelCheckpoint(os.path.join(data_dir('snapshots'),
                                                 'weights.{epoch}-{val_loss:.2f}.h5'),
                                    save_weights_only=True)
    callbacks = [prog_callback, tb_callback, chpt_callback]

    if dataset == 'cifar10':
        to_train_epochs = lambda x: int(x*128/producer.size('train'))
        nb_epoch = to_train_epochs(64000)  # equivalent to 64k iters in paper

        def lr_schedule(epoch):
            print('epoch is: {}'.format(epoch))
            # equivalent to 32k, 48k iters in paper
            bps = [to_train_epochs(x)*epoch_divider for x in [32000, 48000]]
            n = sum([int(epoch > bp) for bp in bps])
            lr = base_lr / math.pow(10, n)
            print('bp is: {}'.format(bps))
            print('n is: {}'.format(n))
            print('learning rate is: {}'.format(lr))
            return lr

        lr_callback = LearningRateScheduler(lr_schedule)
        callbacks.append(lr_callback)

    elif dataset == 'imagenet':
        pass

    else:
        nb_epoch = 1000

    hist = model.fit_generator(producer.flow('train', datagen_train, batch_size=batch_sz),
                               validation_data=producer.flow('val', datagen_test, batch_size=batch_sz*4),
                               nb_val_samples=producer.size('val'),
                               samples_per_epoch=int(producer.size('train')/epoch_divider), nb_epoch=nb_epoch*epoch_divider,
                               callbacks=callbacks)
    print(hist)
    
    eval_res = model.evaluate_generator(producer.flow('test', datagen_test, batch_size=batch_sz), val_samples=producer.size('test'))
    print(eval_res)

def main():
    # dataset = 'imagenet'; layer_count = 18; experiment_name = 'resnet-18'
    dataset = 'cifar10'; layer_count = 32; experiment_name = 'resnet-32'

    model = prepare_model(dataset, layer_count)
    train_model(model, dataset, experiment_name)

if __name__ == '__main__':
    main()
    

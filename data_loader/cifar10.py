import os
import logging

import numpy as np

from keras.utils import np_utils
from keras.datasets import cifar10

log = logging.getLogger(__name__)


class CIFAR10Producer():

    def __init__(self, data_dir):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # split into 45k train, 5k val
        val_sz = 5000; train_sz = X_train.shape[0] - val_sz
        val_idxs = np.random.choice(X_train.shape[0], size=val_sz, replace=False)
        np.save(os.path.join(data_dir('snapshots'), 'val_idxs.npy'), val_idxs)
        train_mask = np.ones(X_train.shape[0], np.bool)
        train_mask[val_idxs] = 0

        self.X_train = X_train[train_mask]
        self.Y_train = Y_train[train_mask]
        self.X_val = X_train[val_idxs]
        self.Y_val = Y_train[val_idxs]
        self.X_test = X_test
        self.Y_test = Y_test

    def train_sample(self):
        return self.X_train

    def flow(self, set, datagen, batch_size):
        set_map = dict(
            train=(self.X_train, self.Y_train),
            val=(self.X_val, self.Y_val),
            test=(self.X_test, self.Y_test)
        )
        assert set in set_map.keys(), 'Set {} not found'.format(set)
        X, Y = set_map[set]

        return datagen.flow(X, Y, batch_size=batch_size)

    def size(self, set):
        set_map = dict(
            train=self.X_train.shape[0],
            val=self.X_val.shape[0],
            test=self.X_test.shape[0]
        )
        assert set in set_map.keys(), 'Set {} not found'.format(set)

        return set_map[set]

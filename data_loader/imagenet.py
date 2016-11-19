import os
import logging

import h5py
import numpy as np
from scipy.io import loadmat

from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import NumpyArrayIterator

log = logging.getLogger(__name__)


class CentreCropImageDataGenerator(ImageDataGenerator):

    def __init__(crop_dims=(224, 224), **kwargs):

        self.crop_dims = crop_dims
        super(CentreCropImageDataGenerator, self).__init__(**kwargs)

    def standardize(self, x):

        x = super(CentreCropImageDataGenerator, self).standardize(x)

        # take centre crop
        im_dims = x.shape[1:3]
        shift_w = (im_dims[0] - self.crop_dims[0]) / 2
        shift_h = (im_dims[1] - self.crop_dims[1]) / 2

        x = x[:,
              shift_w:shift_w + self.crop_dims[0],
              shift_h:shift_h + self.crop_dims[1]]

        return x


class ImageNetArrayIterator(NumpyArrayIterator):

    def __init__(self, X, y, image_data_generator, nb_classes,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):

        self.nb_classes = nb_classes
        super(ImageNetArrayIterator, self).__init__(X, y, image_data_generator,
                                                    batch_size, shuffle, seed,
                                                    dim_ordering,
                                                    save_to_dir, save_prefix, save_format)

    def next(self):
        batch_x, batch_y = super(ImageNetArrayIterator, self).next()
        batch_x = batch_x.astype(np.float32) / 255.0
        batch_y = np_utils.to_categorical(batch_y, self.nb_classes)
        return batch_x, batch_y


class ImageNetProducer():

    def __init__(self, h5_fname, sample_count=5000):

        self.h5_fname = h5_fname
        self.sample_count = sample_count

        self._dset_size_cache = {}

    def train_sample(self):

        with h5py.File(self.h5_fname) as f:
            X_sample = np.array(f['X_train'][:sample_count])

        X_sample = X_sample.astype(np.float32) / 255.0
        return X_sample

    def flow(self, set, datagen, batch_size):
        return datagen.flow(X, Y, batch_size=batch_size)

        with h5py.File(self.h5_fname) as f:
            X = f['X_{}'.format(set)]
            y = f['y_{}'.format(set)]
            assert len(y.shape) == 1

            nb_classes = f['classes'].size

            return ImageNetArrayIterator(X, y, datagen, nb_classes,
                                         batch_size=batch_size, shuffle=False)

            # X_batch = np.array((batch_size, *X.shape[1:]), dtype=X.dtype)
            # y_batch = np.array((batch_size,), dtype=y.dtype)

            # idx = 0
            # while True:
            #     X_batch.fill(0)
            #     y_batch.fill(0)
            #     start_idx = idx
            #     end_idx = idx + batch_size

            #     overflow = (end_idx > X.shape[0])

            #     if overflow:
            #         end_idx = X.shape[0]

            #     sz = end_idx - start_idx
            #     if sz > 0:
            #         X_batch[:sz] = X[start_idx:end_idx]
            #         y_batch[:sz] = y[start_idx:end_idx]

            #     if overflow:
            #         end_idx = batch_size - sz
            #         if end_idx > 0:
            #             X_batch[sz:] = X[:end_idx]
            #             y_batch[sz:] = y[:end_idx]

            #     Y_batch = np_utils.to_categorical(y_batch, nb_classes)
            #     yield X_batch, Y_batch

            #     idx = end_idx

    def size(self, set):

        if set not in self._dset_size_cache:

            with h5py.File(self.h5_fname) as f:
                X = f['X_{}'.format(set)]
                self._dset_size_cache[set] = X.shape[0]

        return self._dset_size_cache[set]

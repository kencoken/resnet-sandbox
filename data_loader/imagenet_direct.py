import os
import logging

import numpy as np
from scipy.io import loadmat

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator, DirectoryIterator

log = logging.getLogger(__name__)


class ImagePathIterator(Iterator):

    def __init__(self, root_directory, image_data_generator, gt_dict,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        '''
        Iterator for images read from disk, given data in dictionary

        gt_dict is dictionary of form: {fname: cls_idx, ...}
        where cls_idx is an index within 'classes'
        '''

        # setup same as for DirectoryIterator

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = root_directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        # set filenames + index
        self.nb_sample = len(gt_dict)
        self.filenames = list(gt_dict.keys())
        self.classes = np.array(list(gt_dict.values()), dtype='int32')
        if classes:
            self.nb_class = len(classes)
            assert np.max(self.classes) < len(classes), 'Ground truth contains class indices which couldn''t be matched to a class!'
        else:
            self.nb_class = np.max(self.classes)+1

        super().__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        return DirectoryIterator.next(self)


class ImageNetProducer():

    def __init__(self, imnet_dir, sample_count=5000, target_size=(224, 224)):
        self.train_images_dir = os.path.join(imnet_dir, 'train')
        self.val_images_dir = os.path.join(imnet_dir, 'val')
        self.devkit_dir = os.path.join(imnet_dir, 'ILSVRC2012_devkit_t12')
        self.target_size = target_size

        datagen = ImageDataGenerator()
        self.train_default_generator = datagen.flow_from_directory(self.train_images_dir, sample_count)

        # read in metadata file and read classes
        meta = loadmat(os.path.join(self.devkit_dir, 'data/meta.mat'))

        num_classes = 1000
        assert all(i == j for i, j in zip(meta['synsets']['ILSVRC2012_ID'][0:num_classes], range(1, num_classes+1))), 'classes are non-contiguous!'
        self.classes = meta['synsets']['WNID'][:num_classes].tolist()

        # read in validation gt (map fname -> index in classes array)
        val_image_paths = os.listdir(self.val_images_dir)
        with open(os.path.join(self.devkit_dir, 'data/ILSVRC2012_validation_ground_truth.txt')) as f:
            val_image_cls_idxs = [int(line.strip()) for line in f]
        assert len(val_image_paths) == len(val_image_cls_idxs), 'Ground truth and num of images in val set differ!'

        self.val_gt_cls_idxs = {image_path: cls_idx for image_path, cls_idx in zip(val_image_paths, val_image_cls_idxs)}

    def train_sample(self):
        X_sample, Y_sample = self.train_default_generator.flow_from_directory(self.train_images_dir,
                                                                              batch_size=self.sample_count,
                                                                              target_size=self.target_size,
                                                                              classes=self.classes).next()
        return X_sample

    def flow(self, set, datagen, batch_size):
        if set == 'train':
            return datagen.flow_from_directory(self.train_images_dir, batch_size=batch_size,
                                               target_size=self.target_size, classes=self.classes)

        elif set == 'val':
            # create validation set iterator - TODO: put into method of ImageDataGenerator class
            val_iterator = ImagePathIterator(
                self.val_images_dir, image_data_generator=datagen, gt_dict=self.val_gt_cls_idxs,
                target_size=self.target_size, classes=self.classes,
                batch_size=batch_size)
            return val_iterator
            
        else:
            raise NotImplemented('Set {} not supported!'.format(set))

    def size(self, set):
        if set == 'train':
            return self.train_default_generator.N

        elif set == 'val':
            return len(self.val_gt_cls_idxs)

        else:
            raise NotImplemented('Set {} not supported!'.format(set))

import os

import pytest
import numpy as np
from scipy.misc import imsave
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from data_loader.producers import ImagePathIterator


def test_ImagePathIterator(tmpdir):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    class_names = ['class_{}'.format(x) for x in range(10)]
    batch_sz = 8
    batch_count = 3
    dset_sz = batch_sz*batch_count

    # gt_dict should be a mapping of impath -> class index
    # where all paths are relative to the root_directory passed
    # to ImagePathIterator on initialization
    gt_dict = {}
    for i in range(dset_sz):
        fname = '{}.jpg'.format(i)
        gt_dict[fname] = y_train[i].item()
        imsave(os.path.join(str(tmpdir), fname), X_train[i])
    fnames = list(gt_dict.keys())
    
    imgen = ImageDataGenerator()
    impath_iterator = ImagePathIterator(str(tmpdir), imgen, gt_dict,
                                        classes=class_names, batch_size=batch_sz,
                                        shuffle=False)

    batch_offset = 0
    for i in range(batch_count):
        batch_x, batch_y = impath_iterator.next()
        assert batch_x.shape[0] == batch_y.shape[0]

        for row_i in range(batch_y.shape[0]):
            fname = fnames[row_i + batch_offset]
            cls_idx = np.flatnonzero(batch_y[row_i])[0].item()
            assert gt_dict[fname] == cls_idx

        batch_offset += batch_sz

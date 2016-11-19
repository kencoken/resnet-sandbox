import os
import math
import random
from itertools import chain, islice
from collections import namedtuple, Counter
from typing import List, Tuple, Iterable, Dict

import numpy as np
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb

import h5py


class opts:
    ilsvrc_dir = '/data/_datasets/ILSVRC2012'
    output_file = '../data/imagenet.h5'
    image_sz = 256
    batch_sz = 128
    debug = False

SetSizes = namedtuple('SetSizes', ['train', 'val'])

def get_ilsvrc_classes(ilsvrc_dir: str):
    meta = loadmat(os.path.join(ilsvrc_dir, 'ILSVRC2012_devkit_t12/data/meta.mat'), squeeze_me=True)

    num_classes = 1000
    assert all(i == j for i, j in zip(meta['synsets']['ILSVRC2012_ID'][0:num_classes], range(1, num_classes+1))), 'classes are non-contiguous!'
    classes = meta['synsets']['WNID'][:num_classes].tolist()

    return classes

def get_set_sizes(ilsvrc_dir: str) -> SetSizes:
    train_images_dir = os.path.join(ilsvrc_dir, 'train')
    synsets = [x for x in os.listdir(train_images_dir) if os.path.isdir(os.path.join(train_images_dir, x))]
    train_count = sum([len(os.listdir(os.path.join(train_images_dir, synset))) for synset in synsets])

    val_count = len(os.listdir(os.path.join(ilsvrc_dir, 'val')))

    return SetSizes(
        train=train_count,
        val=val_count
    )

def get_train_image_paths(ilsvrc_dir: str, classes: List[str]) -> Iterable[Tuple[str, int]]:

    train_images_dir = os.path.join(ilsvrc_dir, 'train')

    synsets = [x for x in os.listdir(train_images_dir) if os.path.isdir(os.path.join(train_images_dir, x))]
    assert set(synsets) == set(classes), 'classes mismatch in train directory!'
    cached_paths = {}

    while len(synsets) > 0:
        synset = random.choice(synsets)
        def create_new():
            paths = os.listdir(os.path.join(train_images_dir, synset))
            random.shuffle(paths)
            return paths
        paths = cached_paths.get(synset) or cached_paths.setdefault(synset, create_new())
        path = paths.pop()
        if len(paths) == 0:
            synsets.remove(synset)
        yield (os.path.join(train_images_dir, synset, path), classes.index(synset))

def get_val_image_paths(ilsvrc_dir: str, classes: List[str]) -> Iterable[Tuple[str, int]]:

    val_images_dir = os.path.join(ilsvrc_dir, 'val')

    val_image_paths = sorted(os.listdir(val_images_dir))
    with open(os.path.join(ilsvrc_dir, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')) as f:
        val_image_cls_idxs = [int(line.strip())-1 for line in f]
    assert len(val_image_paths) == len(val_image_cls_idxs), 'Ground truth and num of images in val set differ!'

    for path, cls_idx in zip(val_image_paths, val_image_cls_idxs):
        yield (os.path.join(val_images_dir, path), cls_idx)

def load_image(impath, imshape=None):

    im = imread(impath)

    was_converted = False
    if len(im.shape) == 2:
        was_converted = True
        # print('image shape is: {}, converting to rgb'.format(im.shape))
        im = gray2rgb(im)

    assert len(im.shape) == 3
    assert im.shape[2] == 3, 'image shape is: {}, was converted: {}'.format(im.shape, was_converted)
    assert im.dtype == np.uint8

    if imshape is not None:
        margin_value = 0

        r1 = float(imshape[0])/imshape[1]
        r2 = float(im.shape[0])/im.shape[1]
        if r2 > r1: # actual image needs to be cropped vertically
            target_width = im.shape[0]/r1
            margin = int((target_width - im.shape[1]) / 2)
            margin_tup = [(0, 0)]*len(im.shape)
            margin_tup[1] = (margin, margin)
            im = np.pad(im, margin_tup, mode='constant', constant_values=margin_value)
        elif r2 < r1: # actual image needs to be cropped horizontally
            target_height = im.shape[1]/r1
            margin = int((target_height - im.shape[0]) / 2)
            margin_tup = [(0,0)]*len(im.shape)
            margin_tup[0] = (margin,margin)
            im = np.pad(im, margin_tup, mode='constant', constant_values=margin_value)
        im = resize(im, imshape, mode='edge', preserve_range=True)
        assert np.max(im) <= 255.0
        assert np.min(im) >= 0.0
        im = np.rint(im).astype(np.uint8)

    return im

def write_data(opts: opts, set_size: int, generator,
               X, y, ids, f) -> None:

    if opts.debug:
        generator = islice(generator, opts.batch_sz*2 + 10)

    batch_count = math.ceil(float(set_size) / opts.batch_sz)

    X_cache = np.zeros((opts.batch_sz, opts.image_sz, opts.image_sz, 3), dtype=np.uint8)
    y_cache = np.zeros((opts.batch_sz,), dtype=np.uint16)
    ids_cache = np.zeros((opts.batch_sz,), dtype=np.uint32)

    for batch_idx, first in enumerate(generator):

        offset = batch_idx*opts.batch_sz
        cls_idxs = Counter()

        X_cache.fill(0)
        y_cache.fill(0)
        ids_cache.fill(0)
        sz = 0

        for idx, (train_path, cls_idx) in enumerate(chain([first], islice(generator, opts.batch_sz-1))):

            im = load_image(train_path, imshape=(opts.image_sz, opts.image_sz))
            id = int(os.path.splitext(os.path.split(train_path)[1])[0].split('_')[-1])

            cls_idxs[cls_idx] += 1
            X_cache[idx] = im
            y_cache[idx] = cls_idx
            ids_cache[idx] = id
            sz += 1

        start_idx = offset
        end_idx = offset + sz

        print('({:.2f}%) {:07d}/{}: {} synsets'.format(float(offset+idx) / set_size * 100.0,
                                                       offset+idx, set_size, len(cls_idxs)))
        if opts.debug:
            print('start_idx: {}, end_idx: {}'.format(start_idx, end_idx))
            print('-----------')

        X[start_idx:end_idx] = X_cache[:sz]
        y[start_idx:end_idx] = y_cache[:sz]
        ids[start_idx:end_idx] = ids_cache[:sz]

## ---

def write_classes(opts: opts, f, classes: List[str]) -> None:

    print('Writing classes...')

    classes_h5 = f.create_dataset('classes',
                                  (len(classes),), dtype="S10")
    for i in range(len(classes)):
        classes_h5[i] = np.string_(classes[i])

def write_train_data(opts: opts, classes: List[str], set_sizes: SetSizes, f) -> None:

    print('Writing training data...')

    X_train = f.create_dataset('X_train',
                               (set_sizes.train, opts.image_sz, opts.image_sz, 3), dtype=np.uint8,
                               chunks=(opts.batch_sz, opts.image_sz, opts.image_sz, 3))
    y_train = f.create_dataset('y_train',
                               (set_sizes.train,), dtype=np.uint16,
                               chunks=(opts.batch_sz*100,))
    ids_train = f.create_dataset('ids_train',
                                 (set_sizes.train,), dtype=np.uint32,
                                 chunks=(opts.batch_sz*100,))
    write_data(opts, set_sizes.train, get_train_image_paths(opts.ilsvrc_dir, classes),
               X_train, y_train, ids_train, f)

def write_val_data(opts: opts, classes: List[str], set_sizes: SetSizes, f) -> None:

    print('Writing validation data...')

    X_val = f.create_dataset('X_val',
                             (set_sizes.val, opts.image_sz, opts.image_sz, 3), dtype=np.uint8,
                             chunks=(opts.batch_sz, opts.image_sz, opts.image_sz, 3))
    y_val = f.create_dataset('y_val',
                             (set_sizes.val,), dtype=np.uint16,
                             chunks=(opts.batch_sz*100,))
    ids_val = f.create_dataset('ids_val',
                               (set_sizes.val,), dtype=np.uint32,
                               chunks=(opts.batch_sz*100,))

    write_data(opts, set_sizes.val, get_val_image_paths(opts.ilsvrc_dir, classes),
               X_val, y_val, ids_val, f)


def main(opts: opts) -> None:

    classes = get_ilsvrc_classes(opts.ilsvrc_dir)
    set_sizes = get_set_sizes(opts.ilsvrc_dir)

    if os.path.exists(opts.output_file):
        os.remove(opts.output_file)

    with h5py.File(opts.output_file) as f:

        write_classes(opts, f, classes)
        write_train_data(opts, classes, set_sizes, f)
        write_val_data(opts, classes, set_sizes, f)
        
            

if __name__ == '__main__':
    main(opts)

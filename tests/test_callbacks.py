import os
from collections import namedtuple

import pytest

from keras_ext.callbacks import ResumableModelCheckpoint


def test_ResumableModelCheckpoint(tmpdir):
    # write test weight files
    epoch_range = range(14)
    val_loss_range = range(279, 265, -1)
    assert len(epoch_range) == len(val_loss_range)
    params = [{'epoch': x[0], 'val_loss': x[1] / 100} for x in zip(epoch_range, val_loss_range)]
    weight_ifo = {'weights.{:04d}-{:.2f}.h5'.format(x['epoch'], x['val_loss']): x
                  for x in params}
    weight_fnames = list(weight_ifo.keys())
    other_fnames = ['some_other_file.txt',
                    'something_else.h5']
    fnames = weight_fnames + other_fnames
    for fname in fnames:
        tmpdir.join(fname).write('.')

    checkpoint_cb = ResumableModelCheckpoint(os.path.join(str(tmpdir), 'weights.{epoch}-{val_loss:.2f}.h5'),
                                             prune_freq=5, save_weights_only=True)

    # ensure checkpoint_ifo is returned correctly
    checkpoint_ifo = checkpoint_cb.get_checkpoint_ifo()
    assert len(checkpoint_ifo) == len(weight_fnames)
    for fname, ifo in checkpoint_ifo.items():
        fname = os.path.split(fname)[1]
        gt_ifo = weight_ifo[fname]
        assert len(ifo) == len(gt_ifo)
        for key, val in ifo.items():
            assert gt_ifo[key] == val

    # ensure get_last_checkpoint works
    last_checkpoint = checkpoint_cb.get_last_checkpoint()
    assert last_checkpoint.initial_epoch == 14
    assert last_checkpoint.filepath == os.path.join(str(tmpdir), weight_fnames[-1])

    # ensure on_epoch_end prunes results correctly
    MockModel = namedtuple('MockModel', ['save_weights'])
    checkpoint_cb.model = MockModel(save_weights=lambda fname, overwrite: print('Saving weights: {}...'.format(fname)))

    checkpoint_cb.on_epoch_end(13, logs={'val_loss': params[-1]['val_loss']})

    expected_epochs = [0, 5, 8, 9, 10, 11, 12, 13]
    expected_weight_fnames = [weight_fname
                              for weight_fname, param in zip(weight_fnames, params)
                              if param['epoch'] in expected_epochs]
    for other_fname in other_fnames:
        os.remove(os.path.join(str(tmpdir), other_fname))
    checkpoint_fnames = os.listdir(str(tmpdir))
    assert set(checkpoint_fnames) == set(expected_weight_fnames)
    

if __name__ == '__main__':
    pytest.main([__file__])

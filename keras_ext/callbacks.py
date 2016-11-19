import os
import re
import logging
from collections import namedtuple

import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)


class ResumableModelCheckpoint(ModelCheckpoint):

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', prune_freq=0):
        super(ResumableModelCheckpoint, self).__init__(filepath, monitor, verbose,
                                                       save_best_only, save_weights_only,
                                                       mode)
        self.prune_freq = prune_freq

    def get_checkpoint_ifo(self):
        ParsedFilepath = namedtuple('ParsedFilepath', ['filepath', 'tokens', 'checkpoint_dir', 'fpath_regex'])

        # compile regex parser for checkpoint filenames if not already cached
        if hasattr(self, 'parsed_filepath_') and self.parsed_filepath_.filepath == self.filepath:
            parsed_filepath = self.parsed_filepath_
        else:
            filepath = self.filepath
            checkpoint_dir, fpath = os.path.split(filepath)

            # find tokens in filepath
            token_regex = re.compile('\{([a-z_]+):?([^\}])*\}')
            tokens = [x.groups()[0] for x in token_regex.finditer(fpath)]

            if 'epoch' not in tokens:
                raise ValueError('ResumableModelCheckpoint requires {epoch} token to be present in filepath pattern!\n'
                                 'Current pattern: {}'.format(filepath))

            # convert tokens to regex for matching with actual filenames
            fpath_pattern = re.escape(token_regex.sub('______', fpath)).replace('______', '([0-9]+\.*[0-9]*)')
            fpath_regex = re.compile(fpath_pattern)

            parsed_filepath = ParsedFilepath(filepath, tokens, checkpoint_dir, fpath_regex)
            self.parsed_filepath_ = parsed_filepath


        # extract token values from matching filenames
        checkpoint_matches = {fpath: parsed_filepath.fpath_regex.match(fpath)
                              for fpath in os.listdir(parsed_filepath.checkpoint_dir)}
        checkpoint_values = {fpath: [int(x) if parsed_filepath.tokens[i] == 'epoch' else float(x)
                                     for i, x in enumerate(match.groups())]
                             for fpath, match in checkpoint_matches.items() if match}
        checkpoint_ifo = {os.path.join(parsed_filepath.checkpoint_dir, fpath): dict(zip(parsed_filepath.tokens, values))
                          for fpath, values in checkpoint_values.items()}

        return checkpoint_ifo

    def get_last_checkpoint(self):
        checkpoint_ifo = self.get_checkpoint_ifo()

        LastCheckpoint = namedtuple('LastCheckpoint', ['initial_epoch', 'filepath'])

        if len(checkpoint_ifo) > 0:
            # find checkpoint with largest epoch
            sorted_checkpoint_ifo = sorted(checkpoint_ifo.items(), key=lambda item: item[1]['epoch'])
            last_filepath, last_ifo = sorted_checkpoint_ifo[-1]
            last_epoch = last_ifo['epoch']
            initial_epoch = last_epoch + 1

            return LastCheckpoint(initial_epoch, last_filepath)
        else:
            return LastCheckpoint(0, None)

    def on_epoch_end(self, epoch, logs={}):
        super(ResumableModelCheckpoint, self).on_epoch_end(epoch, logs)
        if not self.save_best_only and self.prune_freq > 0:
            checkpoint_ifo = self.get_checkpoint_ifo()
            checkpoints_to_prune = [filepath for filepath, ifo in checkpoint_ifo.items()
                                    if ifo['epoch'] + self.prune_freq < epoch and ifo['epoch'] % self.prune_freq != 0]
            if len(checkpoints_to_prune) > 0:
                print('Pruning {} checkpoints: {}'.format(len(checkpoints_to_prune), checkpoints_to_prune))
            for checkpoint in checkpoints_to_prune:
                os.remove(checkpoint)

class TrackEpoch(Callback):

    def __init__(self, filepath):
        super(TrackEpoch, self).__init__()
        self.filepath = filepath
        self.field_name = 'initial_epoch'

    def on_epoch_end(self, epoch, logs={}):
        with open(self.filepath, 'w') as f:
            print('Writing epoch: {}'.format(epoch))
            f.write('{}: {}'.format(self.field_name, epoch+1))

    def get_initial_epoch(self):
        initial_epoch = 0
        if os.path.exists(self.filepath):
            with open(self.filepath) as f:
                line = f.readline().strip()
                if line.startswith('{}:'.format(self.field_name)):
                    line = line[len(self.field_name)+1:].strip()
                    initial_epoch = int(line)
                else:
                    raise RuntimeError('Error reading epoch tracker file!')

        return initial_epoch


class TensorBoardExtra(TensorBoard):

    # def _set_model(self, model):
    #     assert hasattr(model.optimizer, 'lr'), \
    #         'Optimizer must have a "lr" attribute.'

    #     tf.histogram_summary('learning_rate', model.optimizer.lr)
    #     super()._set_model(model)

    def on_epoch_end(self, epoch, logs={}):

        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = self.model.optimizer.lr.eval(session=self.sess).item()
        summary_value.tag = 'learning_rate'
        self.writer.add_summary(summary, epoch)
        
        super().on_epoch_end(epoch, logs)

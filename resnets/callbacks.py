import tensorflow as tf
from keras.callbacks import TensorBoard


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

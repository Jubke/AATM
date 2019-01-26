"""
Custom TensorBoard logger to add settings text summary from a Keras model.
Credit: https://stackoverflow.com/a/52469530/3814081
"""
import tensorflow as tf
from keras.callbacks import TensorBoard

class TensorBoardLogger(TensorBoard):
    """ Extended TensorBoard logger.  """

    def __init__(self, log_dir, settings_str_to_log, **kwargs):
        super(TensorBoardLogger, self).__init__(log_dir, **kwargs)

        self.settings_str = settings_str_to_log

    def on_train_begin(self, logs=None):
        TensorBoard.on_train_begin(self, logs=logs)

        tensor = tf.convert_to_tensor(self.settings_str)
        summary = tf.summary.text("Run_Settings", tensor)

        with  tf.Session() as sess:
            self.writer.add_summary(sess.run(summary))

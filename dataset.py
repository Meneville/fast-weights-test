# ---------------------------------------------------------------------------
# 0. import
# ---------------------------------------------------------------------------
import tensorflow as tf
import numpy as np


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras
    """

    def __init__(self, inputs, labels, batch_sz=32, shuffle=False):
        self.batch_size = batch_sz
        self.inputs = inputs
        self.labels = labels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.inputs))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(indexes)

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.inputs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_index):
        X = self.inputs[list_index]
        Y = self.labels[list_index]
        return X, Y

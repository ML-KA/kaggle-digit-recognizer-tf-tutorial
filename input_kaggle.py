#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


class DataSets(object):
    """Manage train, validation and test sets."""
    pass


class DataSet(object):
    def __init__(self, images, labels=None, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            if labels is not None:
                assert images.shape[0] == labels.shape[0], (
                    "images.shape: %s labels.shape: %s" % (images.shape,
                                                           labels.shape))
        self._num_examples = images.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # assert images.shape[3] == 1
        # images = images.reshape(images.shape[0],
        #                         images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_data_sets(train_path, test_path):
    """
    Parameters
    ----------
    train_path : str
        Path to the train.csv
    test_path : str
        Path to the test.csv

    Returns
    -------
    """
    data_sets = DataSets()
    train_df = pd.read_csv(train_path)
    y_train = train_df[['label']]
    x_train = train_df.ix[:, 1:]
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.10,
                                                      random_state=42)
    data_sets.train = DataSet(x_train, dense_to_one_hot(y_train))
    data_sets.validation = DataSet(x_val, dense_to_one_hot(y_val))

    test_images = pd.read_csv(test_path)
    data_sets.test = DataSet(test_images)

    return data_sets

if __name__ == '__main__':
    read_data_sets('input/train.csv', 'input/test.csv')

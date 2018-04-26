
import os
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from data.data_tf import dataset_utils_tf


class Dataset_TFRecords(object):
    """A simple class for handling data sets."""
    __metaclass__ = ABCMeta

    def __init__(self, name, subset, file_pattern):
        """Initialize dataset using a subset and the path to the data."""
        assert subset in self.available_subsets(), self.available_subsets() #SPLITS_TO_SIZES
        # if self.subset not in SPLITS_TO_SIZES:
        #     raise ValueError('split name %s was not recognized.' % self.subset)
        self.name = name
        self.subset = subset
        self.file_pattern = file_pattern

    @abstractmethod
    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass

    @abstractmethod
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        pass

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']


    def data_files(self):
        """Returns a python list of all (sharded) data subset files.

        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """

        data_files = tf.gfile.Glob(self.file_pattern)
        if not data_files:
            print('No files found for dataset %s/%s at %s' % (self.name,
                                                              self.subset,
                                                              self.file_pattern))

            exit(-1)
        return data_files

    def get_split_slim(self, file_pattern=None, reader=None, tfRecords_dir=None, n_channels=None):
        """Gets a dataset tuple with instructions for reading flowers.

        Args:
          split_name: A train/validation split name.
          dataset_dir: The base directory of the dataset sources.
          file_pattern: The file pattern to use when matching the dataset sources.
            It is assumed that the pattern contains a '%s' string so that the split
            name can be inserted.
          reader: The TensorFlow reader type.

        Returns:
          A `Dataset` namedtuple.

        Raises:
          ValueError: if `split_name` is not a valid train/validation split.
        """
        if not file_pattern:
            file_pattern = self.file_pattern

        self.tfRecords_dir = tfRecords_dir
        self.channels = n_channels

        num_samples = self.num_examples_per_epoch()
        if (not num_samples) or (num_samples<1):
            # TODO: file patter for counting has to be a subset of file_pattern .. how?
            # raise ValueError('Invalid number of samples in tf files')
            file_pattern_for_counting = file_pattern.split('*')[0]
            file_pattern_for_counting = file_pattern_for_counting.split('/')[-1]
            tfrecords_to_count = [os.path.join(self.tfRecords_dir, file) for file in os.listdir(self.tfRecords_dir) \
                                  if (file.startswith(file_pattern_for_counting))]
            num_samples = dataset_utils_tf.number_of_examples_in_tfrecords(tfrecords_to_count)
            print('Found %d records in %s files' %(num_samples, file_pattern_for_counting))

        return dataset_utils_tf.get_split_slim(self.tfRecords_dir, file_pattern, num_samples, self.num_classes(),
                                               self.channels, reader=reader)


    def reader(self):
        """Return a reader for a single entry from the data set.

        See io_ops.py for details of Reader class.

        Returns:
          Reader object that reads the data set.
        """
        return tf.TFRecordReader()

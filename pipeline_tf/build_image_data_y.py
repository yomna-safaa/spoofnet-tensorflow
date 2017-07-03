### YY: inspired from and modified:
## build_image_data.py of inception tf model

# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

where the sub-directory is the unique label associated with these images.

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.


This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'

If you data set involves bounding boxes, please look at build_imagenet_data.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import threading
from datetime import datetime

import tensorflow as tf
from pipeline_tf.image_utils import ImageCoder_TF, process_image_file
from pipeline_tf.paths_namings import get_dataset_paths_and_settings, generate_name_suffix

from data import helpers_dataset

################################################
FLAGS = tf.app.flags.FLAGS

_, TFRecords_DIR, imgs_sub_dirs, csv_files, LABELS_FILE = \
    get_dataset_paths_and_settings(FLAGS.dataset_name)

TRAIN_IMGS_DIR = imgs_sub_dirs['train']
VAL_IMGS_DIR = imgs_sub_dirs['validation']
TEST_IMGS_DIR = imgs_sub_dirs['test']

TRAIN_CSV_FILE = csv_files['train']
VAL_CSV_FILE = csv_files['validation']
TEST_CSV_FILE = csv_files['test']

SHARDS_TRAIN = 2
SHARDS_VAL = 2
SHARDS_TEST = 2
NUM_THREADS = 2
VAL_PERCENT = None

# :YY

tf.app.flags.DEFINE_string('train_directory', TRAIN_IMGS_DIR,  # '/tmp/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', VAL_IMGS_DIR,  # '/tmp/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('test_directory', TEST_IMGS_DIR,  # '/tmp/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('labels_file', LABELS_FILE,  # '',
                           'Labels file')

tf.app.flags.DEFINE_integer('train_shards', SHARDS_TRAIN,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', SHARDS_VAL,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', SHARDS_TEST,
                            'Number of shards in test TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', NUM_THREADS,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_boolean('validation_exists', False,
                            'If already exists a directory containing validation images, or not (for which a subset of the training directory will be used for validation)')
tf.app.flags.DEFINE_boolean('regenerate_label_files', False,
                            'if False => labels files already exists and train/val splits stated in them, do not regenerate labels files')
tf.app.flags.DEFINE_string('train_csv_file', TRAIN_CSV_FILE,  # '',
                           'train_csv_file')
tf.app.flags.DEFINE_string('validation_csv_file', VAL_CSV_FILE,  # '',
                           'validation_csv_file')
tf.app.flags.DEFINE_string('test_csv_file', TEST_CSV_FILE,  # '',
                           'test_csv_file')
tf.app.flags.DEFINE_float('validation_percent', VAL_PERCENT,  # '',
                          'validation_percent')

FLAGS = tf.app.flags.FLAGS


######################################################################
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

######################################################################
def _convert_to_example(filename, image_buffer, label, text, height, width, channels, image_format):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    if channels == 3:
        colorspace = 'RGB'
    elif channels == 1:
        colorspace = 'Grayscale'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


############################################################
def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards, tfRecords_dir, grayscale=False):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        ## YY:
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(tfRecords_dir, output_filename)
        tfrecord_writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            image_buffer, height, width, channels, _ = process_image_file(filename, coder, grayscale, encode_type=FLAGS.encode_type)

            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width, channels, FLAGS.encode_type)
            tfrecord_writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        tfrecord_writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        # shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards, num_threads, tfRecords_dir, grayscale=False):
    ## YY: note: the input lists are already shuffled from before  ... so no need to shuffle again
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder_TF()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards, tfRecords_dir, grayscale)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _process_dataset(name, directory, num_shards, num_threads, tfRecords_dir, labels_file, from_file=False, csv_file=None, test=False):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    ## return SHUFFLED lists of files_absPath and their corresponding labels numbers and labels text (name of label)

    filenames, texts, labels = helpers_dataset.find_image_files(directory, labels_file, from_file, csv_file, test,
                                                                start_labels_at_1=FLAGS.start_labels_at_1)
    mean_pixel = helpers_dataset.calc_mean_pixel(filenames)

    if FLAGS.channels == 3:
        set_grayscale = False
    elif FLAGS.channels == 1:
        set_grayscale = True
    name_suffix = generate_name_suffix(set_grayscale, FLAGS.encode_type)
    name += '-' + name_suffix

    mean_px_file = os.path.join(tfRecords_dir, 'mean_px_' + name + '.txt')
    with open(mean_px_file, 'w') as wf:
        wf.writelines(str(mean_pixel[i]) + '\n' for i in range(0, mean_pixel.shape[0]))

    if num_shards <1:
        assert len(filenames) == len(texts)
        assert len(filenames) == len(labels)
        n = len(filenames)
        max_files_per_shard = 1000
        num_shards = np.ceil(n * 1.0/max_files_per_shard)
        if num_shards ==1:
            num_threads = 1
        else:
            assert not num_shards % num_threads, (
                'Please make the FLAGS.num_threads commensurate with num_shards: ', num_shards)

    _process_image_files(name, filenames, texts, labels, num_shards, num_threads, tfRecords_dir, set_grayscale)

    return mean_pixel

#########################################################
import csv
def read_cats_dogs_datasets(train_dir, test_dir, trainF, valF, testF, validationPercent=0.2):   # set dogs to 0 and cats to 1

    train_dogs = [os.path.join(train_dir, i) for i in os.listdir(train_dir) if 'dog' in i]
    train_cats = [os.path.join(train_dir, i) for i in os.listdir(train_dir) if 'cat' in i]
    test_images = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]

    with open(trainF, 'wb') as tF, open(valF, 'w') as vF:
        tF_wr = csv.writer(tF, delimiter=' ')
        vF_wr = csv.writer(vF, delimiter=' ')

        ## DOGS: train/val
        files = train_dogs
        label = 0
        nAllImages = len(files)
        nTrain = nAllImages - int(round(nAllImages * validationPercent))
        trainImgs = files[0:nTrain]
        valFiles = files[nTrain:]
        for img in trainImgs:
            tF_wr.writerow([img] + [label])
        for img in valFiles:
            vF_wr.writerow([img] + [label])

        ## CATS: train/val
        files = train_cats
        label = 1
        nAllImages = len(files)
        nTrain = nAllImages - int(round(nAllImages * validationPercent))
        trainImgs = files[0:nTrain]
        valFiles = files[nTrain:]
        for img in trainImgs:
            tF_wr.writerow([img] + [label])
        for img in valFiles:
            vF_wr.writerow([img] + [label])

    with open(testF, 'wb') as testFile:
        testFile_wr = csv.writer(testFile, delimiter=' ')
        label = -100 # unknown
        for img in test_images:
            testFile_wr.writerow([img] + [label])

##################################################
def main():
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')
    tfRecords_dir = FLAGS.tfRecords_dir
    if FLAGS.tfRecords_dir is None:
        tfRecords_dir = TFRecords_DIR
    print('Saving results to %s' % tfRecords_dir)

    if not tf.gfile.Exists(tfRecords_dir):
        tf.gfile.MakeDirs(tfRecords_dir)

    ## YY: do it mutiple ways;
    if FLAGS.validation_exists:
        _process_dataset('validation', FLAGS.validation_directory,
                         FLAGS.validation_shards, FLAGS.num_threads, tfRecords_dir, FLAGS.labels_file)
        _process_dataset('train', FLAGS.train_directory,
                         FLAGS.train_shards, FLAGS.num_threads, tfRecords_dir, FLAGS.labels_file)
    else:  ## YY: split
        trainF = FLAGS.train_csv_file
        valF = FLAGS.validation_csv_file
        testF = FLAGS.test_csv_file

        if FLAGS.regenerate_label_files:
            categoriesFile = FLAGS.labels_file + '.again.txt'
            validationPercent = FLAGS.validation_percent

            if FLAGS.dataset_name=='catsDogs':
                testF = FLAGS.test_csv_file
                read_cats_dogs_datasets(FLAGS.train_directory, FLAGS.test_directory, trainF, valF, testF, validationPercent)
            else:
                helpers_dataset.Generate_LabelsFile(FLAGS.train_directory, trainF, valF, categoriesFile, validationPercent)


        _process_dataset('validation', FLAGS.validation_directory,
                         FLAGS.validation_shards, FLAGS.num_threads, tfRecords_dir, FLAGS.labels_file,
                         from_file=True, csv_file=valF)
        _process_dataset('train', FLAGS.train_directory,
                         FLAGS.train_shards, FLAGS.num_threads, tfRecords_dir, FLAGS.labels_file,
                         from_file=True, csv_file=trainF)
        if FLAGS.test_directory is not None:
            if len(FLAGS.test_directory) > 0:
                _process_dataset('test', FLAGS.test_directory,
                                 FLAGS.test_shards, FLAGS.num_threads, tfRecords_dir, FLAGS.labels_file,
                                 from_file=True, csv_file=testF, test=True)


if __name__ == '__main__':
    tf.app.run()

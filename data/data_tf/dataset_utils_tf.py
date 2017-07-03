from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

LABELS_FILENAME = 'labels.txt'


def number_of_examples_in_tfrecords(tfrecords_to_count):
  num_samples = 0
  for tfrecord_file in tfrecords_to_count:
    for record in tf.python_io.tf_record_iterator(tfrecord_file):
      num_samples += 1

  return num_samples


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'r') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names


#################################################################################################
slim = tf.contrib.slim
from pipeline_tf.image_utils import preprocess_image


def load_batch_slim(dataset, batch_size, height, width, preprocessing_name, image_preprocessing_fn,
                    num_readers=1, num_preprocessing_threads=4, per_image_standardization=False,
                    vgg_sub_mean_pixel=None, vgg_resize_side_in=None, vgg_use_aspect_preserving_resize=None,
                    labels_offset=0, is_training=False, allow_smaller_final_batch=False):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    if is_training:
          provider = slim.dataset_data_provider.DatasetDataProvider(
        ## YY: note, shuffle is set to True by default in the Provider init
        dataset,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    else:
      if allow_smaller_final_batch:
          provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, num_epochs=1,
            shuffle=False,
            common_queue_capacity=2 * batch_size,
            common_queue_min=batch_size)
      else:
          provider = slim.dataset_data_provider.DatasetDataProvider(
              dataset,
              shuffle=False,
              common_queue_capacity=2 * batch_size,
              common_queue_min=batch_size)

    [raw_image, label, filename] = provider.get(['image', 'label', 'filename'])
    label -= labels_offset

    image = preprocess_image(preprocessing_name, image_preprocessing_fn, raw_image, height, width,
                             per_image_standardization=per_image_standardization,
                             vgg_sub_mean_pixel=vgg_sub_mean_pixel, vgg_resize_side_in=vgg_resize_side_in,
                             vgg_use_aspect_preserving_resize=vgg_use_aspect_preserving_resize)

    # As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels, filenames =tf.train.batch(
      [image, raw_image, label, filename],
      batch_size=batch_size,
      num_threads=num_preprocessing_threads,
      capacity=5 * batch_size, allow_smaller_final_batch= allow_smaller_final_batch)

    return images, raw_images, labels, filenames


##########################################################################
def get_split_slim(dataset_dir, file_pattern, num_samples, num_classes, channels, reader=None):
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

    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image of varying size.',
        'label': 'A single integer between 0 and 1',
        'filename': 'A string of file name',
    }

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = { ## YY: names of the keys to extract from the tfexamples of the tfrecord
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
    }

    items_to_handlers = { ## YY: names of the tensors to be returned by the decoder (values coming from keys_to_features)
        'image': slim.tfexample_decoder.Image(channels=channels),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if has_labels(dataset_dir):
        labels_to_names = read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=num_classes,
        labels_to_names=labels_to_names)


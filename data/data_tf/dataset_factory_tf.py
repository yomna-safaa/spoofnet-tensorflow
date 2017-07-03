from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data.data_tf.datasets_tf import *

datasets_map = {
    'catsDogs': CatsDogsData,
    'Warsaw': IrisWarsawData,
    'ATVS': IrisATVSData,
    'MobBioFake': IrisMobBioFakeData,
    'ReplayAttack': ReplayAttackData,
}

def get_dataset_y(name, split_name, file_pattern):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  datasetName = datasets_map[name]
  return datasetName(split_name, file_pattern)


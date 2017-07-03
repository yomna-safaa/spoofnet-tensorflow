# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from preprocessing import vgg_preprocessing
from preprocessing import y_vgg_preprocessing, y_preprocessing

slim = tf.contrib.slim

preprocessing_fn_map = {
      'alexnet_v2': y_preprocessing,
      'y_vgg': y_vgg_preprocessing,
      'y_combined': y_preprocessing,
      'vgg': vgg_preprocessing,
}

def get_preprocessing(name, is_training=False):
  """Returns preprocessing_fn(image, height, width, **kwargs).

  Args:
    name: The name of the preprocessing function.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    preprocessing_fn: A function that preprocessing a single image (pre-batch).
      It has the following signature:
        image = preprocessing_fn(image, output_height, output_width, ...).

  Raises:
    ValueError: If Preprocessing `name` is not recognized.
  """


  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image, output_height, output_width, is_training=is_training, **kwargs)

  return preprocessing_fn


def get_img_crop_preprocessing(name):
    """Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
      name: The name of the preprocessing function.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      preprocessing_fn: A function that preprocessing a single image (pre-batch).
        It has the following signature:
          image = preprocessing_fn(image, output_height, output_width, ...).

    Raises:
      ValueError: If Preprocessing `name` is not recognized.
    """

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def process_image_crop_fn(image_crop, **kwargs):
        return preprocessing_fn_map[name].process_image_crop(image_crop, **kwargs)

    return process_image_crop_fn
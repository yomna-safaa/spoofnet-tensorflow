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
import functools

import tensorflow as tf

from nets import alexnet
from nets import cifarnet
from nets import spoofnet_y

slim = tf.contrib.slim

networks_map = {'alexnet_v2': alexnet.alexnet_v2,
                'cifarnet': cifarnet.cifarnet,
                'spoofnet_y': spoofnet_y.spoofnet_y, #YY
                'spoofnet_y_BN': spoofnet_y.spoofnet_y, #YY
                'spoofnet_y_noLRN': spoofnet_y.spoofnet_y_noLRN, #YY
                'spoofnet_y_BN_noLRN': spoofnet_y.spoofnet_y_noLRN, #YY
                'spoofnet_y2_noLRN': spoofnet_y.spoofnet_y2_noLRN, #YY
                'spoofnet_y2_BN_noLRN': spoofnet_y.spoofnet_y2_noLRN,
               }

arg_scopes_map = {'alexnet_v2': alexnet.alexnet_v2_arg_scope,
                  'cifarnet': cifarnet.cifarnet_arg_scope,
                  'spoofnet_y': spoofnet_y.spoofnet_y_arg_scope, #YY
                  'spoofnet_y_BN': spoofnet_y.spoofnet_y_arg_scope_BN, #YY
                  'spoofnet_y_noLRN': spoofnet_y.spoofnet_y_arg_scope, #YY
                  'spoofnet_y_BN_noLRN': spoofnet_y.spoofnet_y_arg_scope_BN, #YY
                  'spoofnet_y2_noLRN': spoofnet_y.spoofnet_y_arg_scope, #YY
                  'spoofnet_y2_BN_noLRN': spoofnet_y.spoofnet_y_arg_scope_BN,
                 }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  @functools.wraps(func)
  # def network_fn(images): ## YY: orig
  def network_fn(images, **kwargs): ## YY: updated 20May2017 to allow passinf of droupout_keep_prob during train
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      # return func(images, num_classes, is_training=is_training) #YY: orig
      return func(images, num_classes, is_training=is_training, **kwargs) ## YY: updated 20May2017 to allow passinf of droupout_keep_prob during train
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn

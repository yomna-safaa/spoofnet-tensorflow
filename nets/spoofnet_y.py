# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the CIFAR-10 model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def spoofnet_y(images, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='SpoofNetY'):
  """Creates a variant of the CifarNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = cifarnet.cifarnet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  end_points = {}

  with tf.variable_scope(scope, 'SpoofNetY', [images, num_classes]):
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net = slim.conv2d(images, 16, [5, 5], scope='conv1') # cifar: 64
      end_points['conv1'] = net
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1') #YY:[2, 2], 2, scope='pool1') ## YY: 20May2017 ,testing to be the same as my model without slim
      end_points['pool1'] = net
      net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
      net = slim.conv2d(net, 64, [5, 5], scope='conv2')
      end_points['conv2'] = net
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')  # YY:[2, 2], 2, scope='pool2') ## YY: 20May2017 ,testing to be the same as my model without slim
      end_points['pool2'] = net
      net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2') ## YY: switch maxpool and lrn from cifar10
      # net = slim.max_pool2d(net, [3, 3], 2, padding='SAME', scope='pool2') #YY:[2, 2], 2, scope='pool2') ## YY: 20May2017 ,testing to be the same as my model without slim
      # end_points['pool2'] = net
      net = slim.flatten(net)
      end_points['Flatten'] = net ## YY: spoonet, no fc layers, only the final fc layer of logits
      # net = slim.fully_connected(net, 384, scope='fc3')
      # end_points['fc3'] = net
      # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
      #                    scope='dropout3')
      # net = slim.fully_connected(net, 192, scope='fc4')
      # end_points['fc4'] = net
      logits = slim.fully_connected(net, num_classes,
                                    biases_initializer=tf.zeros_initializer(),
                                    weights_initializer=trunc_normal(0.0005), #trunc_normal(1/192.0),
                                    weights_regularizer=None, # TODO, use reg)wd) or not for logits?
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='logits')

      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points

def spoofnet_y_noLRN(images, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='SpoofNetY'):
  """Creates a variant of the CifarNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = cifarnet.cifarnet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  end_points = {}

  with tf.variable_scope(scope, 'SpoofNetY', [images, num_classes], reuse=tf.AUTO_REUSE):
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net = slim.conv2d(images, 16, [5, 5], scope='conv1') # cifar: 64
      end_points['conv1'] = net
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1') #YY:[2, 2], 2, scope='pool1') ## YY: 20May2017 ,testing to be the same as my model without slim
      end_points['pool1'] = net
      # net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
      net = slim.conv2d(net, 64, [5, 5], scope='conv2')
      end_points['conv2'] = net
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')  # YY:[2, 2], 2, scope='pool2') ## YY: 20May2017 ,testing to be the same as my model without slim
      end_points['pool2'] = net
      # net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2') ## YY: switch maxpool and lrn from cifar10
      # net = slim.max_pool2d(net, [3, 3], 2, padding='SAME', scope='pool2') #YY:[2, 2], 2, scope='pool2') ## YY: 20May2017 ,testing to be the same as my model without slim
      # end_points['pool2'] = net
      net = slim.flatten(net)
      end_points['Flatten'] = net ## YY: spoonet, no fc layers, only the final fc layer of logits
      # net = slim.fully_connected(net, 384, scope='fc3')
      # end_points['fc3'] = net
      # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
      #                    scope='dropout3')
      # net = slim.fully_connected(net, 192, scope='fc4')
      # end_points['fc4'] = net
      logits = slim.fully_connected(net, num_classes,
                                    biases_initializer=tf.zeros_initializer(),
                                    weights_initializer=trunc_normal(0.0005), #trunc_normal(1/192.0),
                                    weights_regularizer=None, # TODO, use reg)wd) or not for logits?
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='logits')

      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points

def spoofnet_y2_noLRN(images, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='SpoofNetY'):
  """Creates a variant of the CifarNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = cifarnet.cifarnet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  end_points = {}

  with tf.variable_scope(scope, 'SpoofNetY', [images, num_classes]):
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net = slim.conv2d(images, 16, [5, 5], scope='conv1') # cifar: 64
      end_points['conv1'] = net
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1') #YY:[2, 2], 2, scope='pool1') ## YY: 20May2017 ,testing to be the same as my model without slim
      end_points['pool1'] = net
      # net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
      net = slim.conv2d(net, 64, [5, 5], scope='conv2')
      end_points['conv2'] = net
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')  # YY:[2, 2], 2, scope='pool2') ## YY: 20May2017 ,testing to be the same as my model without slim
      end_points['pool2'] = net
      # net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2') ## YY: switch maxpool and lrn from cifar10
      # net = slim.max_pool2d(net, [3, 3], 2, padding='SAME', scope='pool2') #YY:[2, 2], 2, scope='pool2') ## YY: 20May2017 ,testing to be the same as my model without slim
      # end_points['pool2'] = net
      net = slim.flatten(net)
      end_points['Flatten'] = net ## YY: spoonet, no fc layers, only the final fc layer of logits
      net = slim.fully_connected(net, 512, scope='fc')
      end_points['fc'] = net
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout')
      # net = slim.fully_connected(net, 192, scope='fc4')
      # end_points['fc4'] = net
      logits = slim.fully_connected(net, num_classes,
                                    biases_initializer=tf.zeros_initializer(),
                                    weights_initializer=trunc_normal(0.0005), #trunc_normal(1/192.0),
                                    weights_regularizer=None, # TODO, use reg)wd) or not for logits?
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='logits')

      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points


spoofnet_y.default_image_size = 112


def spoofnet_y_arg_scope_BN(weight_decay=0.0004,
                         use_batch_norm=True,
                         batch_norm_decay=0.9, #0.9997, ## YY: dec to 0.9 based on hint from https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-261041193
                         batch_norm_epsilon=0.001
                         ):
  """Defines the default cifarnet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  ## YY: add  batch normalization
  batch_norm_params = {
    # Decay for the moving averages.
    'decay': batch_norm_decay,
    # epsilon to prevent 0s in variance.
    'epsilon': batch_norm_epsilon,
    # collection containing update_ops.
    # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
    # # YY: hint from: https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-235928564:
    # # But what it is important is that either you pass updates_collections=None so
    # # the moving_mean and moving_variance are updated in-place, otherwise you will need gather the update_ops and make sure they are run.
    # 'updates_collections': None
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  with slim.arg_scope(
      [slim.conv2d],
      weights_initializer=tf.truncated_normal_initializer(stddev=5e-2), #TODO: or: weights_initializer=slim.variance_scaling_initializer(), as inception/vgg/resnet
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=normalizer_params
    ):
    # with slim.arg_scope(
    #     [slim.fully_connected],
    #     biases_initializer=tf.constant_initializer(0.1),
    #     weights_initializer=trunc_normal(0.04),
    #     weights_regularizer=slim.l2_regularizer(weight_decay),
    #     activation_fn=tf.nn.relu):
    with slim.arg_scope([slim.max_pool2d], padding='SAME') as sc:
      return sc


def spoofnet_y_arg_scope(weight_decay=0.0004):
  """Defines the default cifarnet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d],
      weights_initializer=tf.truncated_normal_initializer(stddev=5e-2), #TODO: or: weights_initializer=slim.variance_scaling_initializer(), as inception/vgg/resnet
      # weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu
    ):
    # with slim.arg_scope(
    #     [slim.fully_connected],
    #     biases_initializer=tf.constant_initializer(0.1),
    #     weights_initializer=trunc_normal(0.04),
    #     weights_regularizer=slim.l2_regularizer(weight_decay),
    #     activation_fn=tf.nn.relu):
    with slim.arg_scope([slim.max_pool2d], padding='SAME') as sc:
      return sc

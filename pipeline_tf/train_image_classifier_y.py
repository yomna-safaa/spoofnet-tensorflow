from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
# sys.path.insert(0,"/media/yomna/YY32/Google_Drive/PhD/PhD_Research_Projects/antispoof_pipeline_tf_min_July2017")
sys.path.insert(0,"H:\\Google_Drive\\PhD\\PhD_Research_Projects\\antispoof_pipeline_tf_min_July2017")

import math

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

import tensorflow as tf

from data.data_tf.dataset_utils_tf import load_batch_slim
from pipeline_tf import evaluation_y
from pipeline_tf import paths_namings, y_flags  # YY
from pipeline_tf import slim_learning_y
from nets import nets_factory
from preprocessing import preprocessing_factory

from data.data_tf import dataset_factory_tf

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """

  num_batches_per_epoch = num_samples_per_epoch / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch *
                    FLAGS.num_epochs_per_decay)

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.initial_learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.initial_learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _add_variables_summaries(learning_rate):
  summaries = []
  for variable in slim.get_model_variables():
    summaries.append(tf.summary.histogram(variable.op.name, variable))
  summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
  return summaries


def _get_init_fn(checkpoints_dir_y):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  # checkpoints_dir_y = namings.generate_checkpoints_dir(FLAGS, image_size)
  if tf.train.latest_checkpoint(checkpoints_dir_y): #FLAGS.checkpoints_dir): # YY
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % checkpoints_dir_y) # YY: FLAGS.checkpoints_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


########################################################################
def get_logits_and_valid_ops(images, labels, network_fn, one_hot=False, scope='', batch_summ=False):
    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(images)

    top_1_op, acc_op, n_correct, n_all, labels_not_one_hot = evaluation_y.get_eval_ops(logits, labels, one_hot=one_hot, scope=scope, calc_accuracy=batch_summ)
    eval_ops = [top_1_op, acc_op, n_correct, n_all, labels_not_one_hot]
    if batch_summ:
        with tf.name_scope(scope):
            tf.summary.scalar('accInBatch', acc_op)
        names_to_values, names_to_updates = None, None
    else:
        names_to_values, names_to_updates = evaluation_y.get_eval_ops_slim(logits, labels, '',
                                                                           scope=scope + '/Streaming')
        with tf.name_scope(scope + '/Streaming'):
            for name, value in names_to_values.items():
                summary_name = name
                op = tf.summary.scalar(summary_name, value, collections=[])
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    return logits, end_points, names_to_values, names_to_updates, eval_ops


########################################################################
def get_images_and_labels(dataset_split_name, preprocessing_name, image_preprocessing_fn, train_image_size,
                           tfRecords_dir, is_training=False, allow_smaller_final_batch=False, batch_size=None):
    pattern = paths_namings.file_pattern_tfrecords(FLAGS, tfRecords_dir, dataset_split_name)
    dataset_Ob = dataset_factory_tf.get_dataset_y(FLAGS.dataset_name, dataset_split_name, file_pattern=pattern)
    dataset = dataset_Ob.get_split_slim(tfRecords_dir=tfRecords_dir, n_channels=FLAGS.channels)

    if not batch_size:
        batch_size = FLAGS.batch_size
    images, raw_images, labels, filenames = load_batch_slim(dataset, batch_size, train_image_size, train_image_size,
                                                            preprocessing_name, image_preprocessing_fn,
                                                            num_readers=FLAGS.num_readers, num_preprocessing_threads=FLAGS.num_preprocessing_threads,
                                                            per_image_standardization=FLAGS.per_image_standardization,
                                                            vgg_sub_mean_pixel=FLAGS.vgg_sub_mean_pixel,
                                                            vgg_resize_side_in=FLAGS.vgg_resize_side,
                                                            vgg_use_aspect_preserving_resize=FLAGS.vgg_use_aspect_preserving_resize,
                                                            labels_offset=FLAGS.labels_offset,
                                                            is_training=is_training, allow_smaller_final_batch= allow_smaller_final_batch)
    return images, labels, filenames, dataset.num_samples


########################################################################
def train_step_fn_y(sess, train_op, global_step, train_step_kwargs):
    total_loss, should_stop, np_global_step = slim_learning_y.train_step_y(sess, train_op, global_step, train_step_kwargs)

    if 'should_eval_train' in train_step_kwargs and FLAGS.validation_every_n_steps>0:
        if sess.run(train_step_kwargs['should_eval_train']):
            tf.logging.info('--------- Starting evaluation on Training set:')
            summ_writer = train_step_kwargs['summary_writer']
            accuracyy, _, _, _, _, _, _, _ = evaluation_y.evaluate_loop_y(sess, train_step_kwargs['num_batches_train'], FLAGS.batch_size, train_step_kwargs['eval_ops_train'])

            tag_name = 'Evals_Train/whole_set/accuracy'
            summary_str = tf.Summary()
            summary_str.value.add(tag=tag_name, simple_value=accuracyy)
            summ_writer.add_summary(summary_str, np_global_step)
            tf.logging.info('------ @global step %d,  %s: [%.4f]', np_global_step, tag_name, accuracyy)

    if 'should_eval_val' in train_step_kwargs and FLAGS.test_every_n_steps > 0:
        if sess.run(train_step_kwargs['should_eval_val']):
            # tf.logging.info('global step %d: loss = %.4f (%.2f sec/step)', np_global_step, total_loss, time_elapsed)
            tf.logging.info('--------- Starting evaluation on validation set:')
            summ_writer = train_step_kwargs['summary_writer']
            accuracyy, _, _, _, _, _, _, _ = evaluation_y.evaluate_loop_y(sess, train_step_kwargs['num_batches_val'], FLAGS.batch_size, train_step_kwargs['eval_ops_val'])

            tag_name = 'Evals_Val/whole_set/accuracy'
            summary_str = tf.Summary()
            summary_str.value.add(tag=tag_name, simple_value=accuracyy)
            summ_writer.add_summary(summary_str, np_global_step)
            tf.logging.info('------ @global step %d,  %s: [%.4f]', np_global_step, tag_name, accuracyy)

    return total_loss, should_stop

########################################################################3
def main(_):
  tfRecords_dir = FLAGS.tfRecords_dir
  if (FLAGS.tfRecords_dir is None) or (FLAGS.use_placeholders):
        _, tfRecords_dir, imgs_sub_dirs, csv_files, categories_file = paths_namings.get_dataset_paths_and_settings(
            FLAGS.dataset_name)

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # Create global_step
    global_step = tf.train.create_global_step()

    ######################
    # Select the dataset #
    ######################
    pattern = paths_namings.file_pattern_tfrecords(FLAGS, tfRecords_dir, FLAGS.dataset_split_name)
    dataset_Ob = dataset_factory_tf.get_dataset_y(FLAGS.dataset_name, FLAGS.dataset_split_name, file_pattern=pattern)
    dataset = dataset_Ob.get_split_slim(tfRecords_dir=tfRecords_dir, n_channels=FLAGS.channels)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(FLAGS.model_name,
                                             num_classes=(dataset.num_classes - FLAGS.labels_offset),
                                             weight_decay=FLAGS.weight_decay, is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    train_image_size = FLAGS.image_size or network_fn.default_image_size
    images, _, labels, _ = load_batch_slim(dataset, FLAGS.batch_size, train_image_size, train_image_size,
                                             preprocessing_name, image_preprocessing_fn,
                                             num_readers=FLAGS.num_readers, num_preprocessing_threads=FLAGS.num_preprocessing_threads,
                                             per_image_standardization=FLAGS.per_image_standardization,
                                             vgg_sub_mean_pixel=FLAGS.vgg_sub_mean_pixel,
                                             vgg_resize_side_in=FLAGS.vgg_resize_side,
                                             vgg_use_aspect_preserving_resize=FLAGS.vgg_use_aspect_preserving_resize,
                                             labels_offset=FLAGS.labels_offset,
                                             is_training=True)

    #############
    logits, end_points, _, _, eval_ops = get_logits_and_valid_ops(images, labels, network_fn, one_hot=False,
                                                                  batch_summ=True, scope='Evals_Train/Batch')

    #############################
    # Specify the loss function #
    #############################
    labels_one_hot = slim.one_hot_encoding(labels, dataset.num_classes - FLAGS.labels_offset)
    if 'AuxLogits' in end_points:
        tf.losses.softmax_cross_entropy(
            logits=end_points['AuxLogits'], onehot_labels=labels_one_hot,
            weights=0.4, scope='aux_loss')
    tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels_one_hot, weights=1.0)

    #############
    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for end_points.
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, ''):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #########################################
    # Configure the optimization procedure. #
    #########################################
    learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
    optimizer = _configure_optimizer(learning_rate)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #############
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, '')  # YY: first_clone_scope)

    if FLAGS.moving_average_decay: # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    ############# Loss and Optimizing
    total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well
    clones_gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)

    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    train_tensor = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
    train_op_y = train_tensor

    ###########################
    ###########################
    checkpoints_dir_y = paths_namings.generate_checkpoints_dir(FLAGS, train_image_size)
    checkpoints_dir_y = checkpoints_dir_y + '_slim/pr_' + preprocessing_name
    print (checkpoints_dir_y)

    #################################################################################
    network_fn_eval = nets_factory.get_network_fn(FLAGS.model_name,
                                                  num_classes=(dataset.num_classes - FLAGS.labels_offset),
                                                  is_training=False)

    ## ==========
    num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
    _, _, names_to_values, names_to_updates, eval_ops_tr = get_logits_and_valid_ops(images, labels, network_fn_eval,
                                                                                        scope='Evals_Train')
    ## ==========
    images_val, labels_val, _, num_samples_val = get_images_and_labels('validation', preprocessing_name, image_preprocessing_fn,
                                                        train_image_size, tfRecords_dir, is_training=False)
    num_batches_valid = math.ceil(num_samples_val / float(FLAGS.batch_size))
    _, _, names_to_values_valid, names_to_updates_valid, eval_ops_v = get_logits_and_valid_ops(images_val, labels_val, network_fn_eval,
                                                                                       scope='Evals_Val')

    ########################################################################################
    # Add the summaries that contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, ''))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    ########################################################################################
    with tf.name_scope('train_step_eval'):
        train_step_kwargs_extra = {}
        train_step_kwargs_extra['should_eval_train'] = math_ops.equal(
            math_ops.mod(global_step, FLAGS.validation_every_n_steps), 0)
        train_step_kwargs_extra['should_eval_val'] = math_ops.equal(
            math_ops.mod(global_step, FLAGS.test_every_n_steps), 0)
        train_step_kwargs_extra['num_batches_train'] = num_batches
        train_step_kwargs_extra['num_batches_val'] = num_batches_valid
        train_step_kwargs_extra['eval_ops_slim_train'] = list(names_to_updates.values())
        train_step_kwargs_extra['eval_ops_slim_val'] = list(names_to_updates_valid.values())
        train_step_kwargs_extra['stream_acc_slim_train'] = names_to_values['Accuracy']
        train_step_kwargs_extra['stream_acc_slim_val'] = names_to_values_valid['Accuracy']
        train_step_kwargs_extra['eval_ops_train'] = eval_ops_tr
        train_step_kwargs_extra['eval_ops_val'] = eval_ops_v

    ########################################################################################
    tf.logging.info(' ********** Starting Training %s' % checkpoints_dir_y)
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    slim_learning_y.train_y( # slim.learning.train(
        train_op_y,
        logdir=checkpoints_dir_y,
        is_chief=True,
        init_fn=_get_init_fn(checkpoints_dir_y),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        eval_ops=list(names_to_updates.values()), # YY
        num_evals=num_batches,
        eval_ops_valid=list(names_to_updates_valid.values()),  # YY
        num_evals_valid=num_batches_valid,
        session_config=session_config,
        train_step_fn=train_step_fn_y,
        train_step_kwargs_extra=train_step_kwargs_extra
        )
    tf.logging.info(' ********** Finished Training %s' % checkpoints_dir_y)


if __name__ == '__main__':
  tf.app.run()

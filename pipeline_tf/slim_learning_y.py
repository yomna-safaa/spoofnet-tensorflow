## YY: modified from source: tensorflow-master/tensorflow/contrib/slim/python/slim/learning.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from pipeline_tf import evaluation_y
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import training_util

__all__ = [
    'train_step_y', 'train_y'
]


_USE_GLOBAL_STEP = 0

def _wait_for_step(sess, global_step, step):
  """Wait till the global step has reached at least 'step'.

  Args:
    sess: A session.
    global_step: A Tensor.
    step: Int.  The global step to reach.
  """
  while True:
    if training_util.global_step(sess, global_step) >= step:
      break
    time.sleep(1.0)


def train_step_y(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  # logging.info('Global step value in train_step_y: ' + str(np_global_step))
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                   np_global_step, total_loss, time_elapsed)

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop, np_global_step

_USE_DEFAULT = 0

def train_y(train_op,
            logdir,
            train_step_fn=train_step_y,
            train_step_kwargs=_USE_DEFAULT,
            train_step_kwargs_extra=None,  # YY: i added these to be able to pass extra args for evaluation
            log_every_n_steps=1,
            eval_ops=None,  # :YY
            num_evals=0,  # YY
            eval_ops_valid=None,  # :YY
            num_evals_valid=0,  # YY
            graph=None,
            master='',
            is_chief=True,
            global_step=None,
            number_of_steps=None,
            init_op=_USE_DEFAULT,
            init_feed_dict=None,
            local_init_op=_USE_DEFAULT,
            init_fn=None,
            ready_op=_USE_DEFAULT,
            summary_op=_USE_DEFAULT,
            save_summaries_secs=600,
            summary_writer=_USE_DEFAULT,
            startup_delay_steps=0,
            saver=None,
            save_interval_secs=600,
            # sync_optimizer=None, # :YY
            session_config=None,
            trace_every_n_steps=None):
  """Runs a training loop using a TensorFlow supervisor.

  When the sync_optimizer is supplied, gradient updates are applied
  synchronously. Otherwise, gradient updates are applied asynchronous.

  Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where training logs are written to. If None, model
      checkpoints and summaries will not be written.
    train_step_fn: The function to call in order to execute a single gradient
      step. The function must have take exactly four arguments: the current
      session, the `train_op` `Tensor`, a global step `Tensor` and a dictionary.
    train_step_kwargs: A dictionary which is passed to the `train_step_fn`. By
      default, two `Boolean`, scalar ops called "should_stop" and "should_log"
      are provided.
    log_every_n_steps: The frequency, in terms of global steps, that the loss
      and global step and logged.
    graph: The graph to pass to the supervisor. If no graph is supplied the
      default graph is used.
    master: The address of the tensorflow master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    global_step: The `Tensor` representing the global step. If left as `None`,
      then slim.variables.get_or_create_global_step() is used.
    number_of_steps: The max number of gradient steps to take during training.
      If the value is left as None, training proceeds indefinitely.
    init_op: The initialization operation. If left to its default value, then
      the session is initialized by calling `tf.global_variables_initializer()`.
    init_feed_dict: A feed dictionary to use when executing the `init_op`.
    local_init_op: The local initialization operation. If left to its default
      value, then the session is initialized by calling
      `tf.local_variables_initializer()` and `tf.tables_initializer()`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    ready_op: Operation to check if the model is ready to use. If left to its
      default value, then the session checks for readiness by calling
      `tf.report_uninitialized_variables()`.
    summary_op: The summary operation.
    save_summaries_secs: How often, in seconds, to save summaries.
    summary_writer: `SummaryWriter` to use.  Can be `None`
      to indicate that no summaries should be written. If unset, we
      create a SummaryWriter.
    startup_delay_steps: The number of steps to wait for before beginning. Note
      that this must be 0 if a sync_optimizer is supplied.
    saver: Saver to save checkpoints. If None, a default one will be created
      and used.
    save_interval_secs: How often, in seconds, to save the model to `logdir`.
    sync_optimizer: an instance of tf.train.SyncReplicasOptimizer. If the
      argument is supplied, gradient updates will be synchronous. If left as
      `None`, gradient updates will be asynchronous.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    trace_every_n_steps: produce and save a `Timeline` in Chrome trace format
      and add it to the summaries every `trace_every_n_steps`. If None, no trace
      information will be produced or saved.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `train_op` is empty or if `startup_delay_steps` is
      non-zero when `sync_optimizer` is supplied, if `number_of_steps` is
      negative, or if `trace_every_n_steps` is not `None` and no `logdir` is
      provided.
  """
  if train_op is None:
    raise ValueError('train_op cannot be None.')

  if logdir is None:
    if summary_op != _USE_DEFAULT:
      raise ValueError('Cannot provide summary_op because logdir=None')
    if saver is not None:
      raise ValueError('Cannot provide saver because logdir=None')
    if trace_every_n_steps is not None:
      raise ValueError('Cannot provide trace_every_n_steps because '
                       'logdir=None')

  if number_of_steps is not None and number_of_steps <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  graph = graph or ops.get_default_graph()
  with graph.as_default():
    if global_step is None:
      global_step = variables.get_or_create_global_step()
    saver = saver or tf_saver.Saver()

    with ops.name_scope('init_ops'):
      if init_op == _USE_DEFAULT:
        init_op = tf_variables.global_variables_initializer()

      if ready_op == _USE_DEFAULT:
        ready_op = tf_variables.report_uninitialized_variables()

      if local_init_op == _USE_DEFAULT:
        ## YY: I separated them,
        y_local_init_op =tf_variables.local_variables_initializer()
        ## end YY
        local_init_op = control_flow_ops.group(
            y_local_init_op, # YY # tf_variables.local_variables_initializer(),
            data_flow_ops.tables_initializer())

      ready_for_local_init_op = None

    if summary_op == _USE_DEFAULT:
      summary_op = summary.merge_all()

    if summary_writer == _USE_DEFAULT:
      summary_writer = supervisor.Supervisor.USE_DEFAULT

    if train_step_kwargs == _USE_DEFAULT:
      with ops.name_scope('train_step'):
        train_step_kwargs = {}

        if number_of_steps:
          should_stop_op = math_ops.greater_equal(global_step, number_of_steps)
        else:
          should_stop_op = constant_op.constant(False)
        train_step_kwargs['should_stop'] = should_stop_op
        train_step_kwargs['should_log'] = math_ops.equal(
            math_ops.mod(global_step, log_every_n_steps), 0)
        if is_chief and trace_every_n_steps is not None:
          train_step_kwargs['should_trace'] = math_ops.equal(
              math_ops.mod(global_step, trace_every_n_steps), 0)
          train_step_kwargs['logdir'] = logdir
        ## YY:
        if train_step_kwargs_extra is not None:
            for name in train_step_kwargs_extra:
                train_step_kwargs[name] = train_step_kwargs_extra[name]
        ## :YY

  sv = supervisor.Supervisor(
      graph=graph,
      is_chief=is_chief,
      logdir=logdir,
      init_op=init_op,
      init_feed_dict=init_feed_dict,
      local_init_op=local_init_op,
      ready_for_local_init_op=ready_for_local_init_op,
      ready_op=ready_op,
      summary_op=summary_op,
      summary_writer=summary_writer,
      global_step=global_step,
      saver=saver,
      save_summaries_secs=save_summaries_secs,
      save_model_secs=save_interval_secs,
      init_fn=init_fn)

  if summary_writer is not None:
    train_step_kwargs['summary_writer'] = sv.summary_writer
    train_step_kwargs['local_init_op'] = y_local_init_op # tf_variables.local_variables_initializer() #local_init_op  ## YY I added this

  should_retry = True
  while should_retry:
    try:
      should_retry = False
      with sv.managed_session(
          master, start_standard_services=False, config=session_config) as sess:
        logging.info('Starting Session.')
        if is_chief:
          if logdir:
            sv.start_standard_services(sess)
        elif startup_delay_steps > 0:
          _wait_for_step(sess, global_step,
                         min(startup_delay_steps, number_of_steps or
                             sys.maxint))
        sv.start_queue_runners(sess)
        logging.info('Starting Queues.')
        try:
          while not sv.should_stop():
            total_loss, should_stop = train_step_fn(
                sess, train_op, global_step, train_step_kwargs)
            if should_stop:
              logging.info('Stopping Training.')
              break
        except errors.OutOfRangeError:
          # OutOfRangeError is thrown when epoch limit per
          # tf.train.limit_epochs is reached.
          logging.info('Caught OutOfRangeError. Stopping Training.')
        if logdir and sv.is_chief:
          logging.info('Finished training! Saving model to disk.')
          sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

          ## YY: evaluating here
          if eval_ops is not None:
              logging.info('********* Starting evaluation on Training set, at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                                     time.gmtime()))
              evaluation_y.evaluate_loop_slim_streaming_metrics(sess, num_evals, eval_ops)
              summary_str = sess.run(summary_op)
              sv.summary_computed(sess, summary_str)
              logging.info('********* Finished evaluation on Training set, at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                                 time.gmtime()))
          ## YY: starting evaluation on validation set if exist:
          if eval_ops_valid is not None:
              logging.info('********* Starting evaluation on Validation set, at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                                                                time.gmtime()))
              np_global_step = training_util.global_step(sess, global_step)
              evaluation_y.evaluate_loop_slim_streaming_metrics(sess, num_evals_valid, eval_ops_valid)
              summary_str = sess.run(summary_op)
              sv.summary_writer.add_summary(summary_str, np_global_step)
              logging.info('********* Finished evaluation on Validation set, at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                                                                time.gmtime()))
          # :YY

    except errors.AbortedError:
      # Always re-run on AbortedError as it indicates a restart of one of the
      # distributed tensorflow servers.
      logging.info('Retrying training!')
      should_retry = True

  return total_loss


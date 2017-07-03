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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import tensorflow as tf
from data import helpers_dataset
from data.data_tf.dataset_utils_tf import load_batch_slim
from pipeline_tf import evaluation_y
from pipeline_tf import paths_namings, y_flags
from nets import nets_factory
from preprocessing import preprocessing_factory
from pipeline_tf.train_image_classifier_y import get_images_and_labels, get_logits_and_valid_ops

from data.data_tf import dataset_factory_tf

slim = tf.contrib.slim


tf.app.flags.DEFINE_integer(
    'eval_batch_size', 30, '')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'dataset_split_name_y', 'validation',
    'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_split_name_y2', None,
    'The name of the train/test split.')

tf.app.flags.DEFINE_boolean('use_slim_stream_eval', False,
                            '')

tf.app.flags.DEFINE_boolean('use_placeholders', False,
                            '')

tf.app.flags.DEFINE_boolean('oversample_at_eval', False,
                            '')

tf.app.flags.DEFINE_string(
    'input_csv_file', None,
    'The name of the train/test split.')

FLAGS = tf.app.flags.FLAGS


########################################################################
def main(_):
  use_placeholders = FLAGS.use_placeholders
  eval_batch_size = FLAGS.eval_batch_size

  tfRecords_dir = FLAGS.tfRecords_dir
  if (FLAGS.tfRecords_dir is None) or (use_placeholders):
      _, tfRecords_dir, imgs_sub_dirs, csv_files, categories_file = paths_namings.get_dataset_paths_and_settings(
          FLAGS.dataset_name)
      tf.logging.info('Data Dir: %s\nCategoriesFile: %s', tfRecords_dir, categories_file)

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    pattern = paths_namings.file_pattern_tfrecords(FLAGS, tfRecords_dir, FLAGS.dataset_split_name_y)
    dataset_Ob = dataset_factory_tf.get_dataset_y(FLAGS.dataset_name, FLAGS.dataset_split_name_y, file_pattern=pattern)
    dataset = dataset_Ob.get_split_slim(tfRecords_dir=tfRecords_dir, n_channels=FLAGS.channels)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(FLAGS.model_name,
                                             num_classes=(dataset.num_classes - FLAGS.labels_offset),
                                             is_training=False)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.image_size or network_fn.default_image_size

    if use_placeholders:
        images_pl = tf.placeholder(tf.float32, [None, eval_image_size, eval_image_size, FLAGS.channels])
        if (FLAGS.model_name == 'cifarnet') or (FLAGS.model_name == 'spoofnet_y'):
            logits_pl, end_points_pl = network_fn(images_pl, dropout_keep_prob=1)
        else:
            logits_pl, end_points_pl = network_fn(images_pl)

        if FLAGS.input_csv_file:
            filenames, _, labels = helpers_dataset.read_image_files_from_csv(FLAGS.input_csv_file, delimiter=',')
        else:
            filenames, _, labels = helpers_dataset.read_image_files_from_csv(csv_files[FLAGS.dataset_split_name_y],
                                                                             imgs_sub_dirs[FLAGS.dataset_split_name_y], categories_file)
        labels -= FLAGS.labels_offset
        probabilities_op = end_points_pl['Predictions']

        classifier_pl = evaluation_y.Classifier_PL(eval_image_size, FLAGS.channels, preprocessing_name, encode_type=FLAGS.encode_type,
                                                   oversample=FLAGS.oversample_at_eval,
                                                   per_image_standardization=FLAGS.per_image_standardization,
                                                   vgg_sub_mean_pixel=FLAGS.vgg_sub_mean_pixel,
                                                   vgg_resize_side_in=FLAGS.vgg_resize_side,
                                                   vgg_use_aspect_preserving_resize=FLAGS.vgg_use_aspect_preserving_resize
                                                   )

    else:
        images, raw_images, labels_t, filenames_op = load_batch_slim(dataset, eval_batch_size, eval_image_size,
                                                                eval_image_size,
                                                                preprocessing_name, image_preprocessing_fn,
                                                                num_preprocessing_threads=FLAGS.num_preprocessing_threads,
                                                                per_image_standardization=FLAGS.per_image_standardization,
                                                                vgg_sub_mean_pixel=FLAGS.vgg_sub_mean_pixel,
                                                                vgg_resize_side_in=FLAGS.vgg_resize_side,
                                                                vgg_use_aspect_preserving_resize=FLAGS.vgg_use_aspect_preserving_resize,
                                                                labels_offset=FLAGS.labels_offset,
                                                                is_training=False, allow_smaller_final_batch=True)
        _, end_points, names_to_values, names_to_updates, eval_ops_y = get_logits_and_valid_ops(images, labels_t,
                                                                                                network_fn,
                                                                                                scope='Evaluation/' + FLAGS.dataset_split_name_y)
        eval_ops_slim = list(names_to_updates.values())
        accuracy_value = names_to_values['Accuracy']
        probabilities_op = end_points['Predictions']

        num_batches = int(math.ceil(dataset.num_samples / float(eval_batch_size)))
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches

    ################### YY: testing another set:
    # ==========
    if FLAGS.dataset_split_name_y2 is not None:
        if use_placeholders:
            filenames_2, _, labels_2 = helpers_dataset.read_image_files_from_csv(csv_files[FLAGS.dataset_split_name_y2],
                                                                                 imgs_sub_dirs[FLAGS.dataset_split_name_y2],
                                                                                 categories_file)
            labels_2 -= FLAGS.labels_offset
        else:
            images_2, labels_t2, filenames_op_2, num_samples_2 = get_images_and_labels(FLAGS.dataset_split_name_y2, preprocessing_name,
                                                                        image_preprocessing_fn,
                                                                        eval_image_size, tfRecords_dir,
                                                                        is_training=False, allow_smaller_final_batch=True,
                                                                                       batch_size=eval_batch_size)
            num_batches_2 = int(math.ceil(num_samples_2 / float(eval_batch_size)))
            if FLAGS.max_num_batches:
                num_batches_2 = FLAGS.max_num_batches
            _, end_points_2, names_to_values_2, names_to_updates_2, eval_ops_y_2 = get_logits_and_valid_ops(images_2, labels_t2,
                                                                                                 network_fn,
                                                                                                 scope='Evaluation/'+FLAGS.dataset_split_name_y2)
            eval_ops_slim_2 = list(names_to_updates_2.values())
            accuracy_value_2 = names_to_values_2['Accuracy']
            probabilities_op_2 = end_points_2['Predictions']

    ## ==========
    if FLAGS.moving_average_decay:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, tf_global_step)
        variables_to_restore = variable_averages.variables_to_restore(
            slim.get_model_variables())
        variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    ##############################################################
    checkpoints_dir_y = paths_namings.generate_checkpoints_dir(FLAGS, eval_image_size)
    checkpoints_dir_y = checkpoints_dir_y + '_slim/pr_' +preprocessing_name
    if tf.gfile.IsDirectory(checkpoints_dir_y):
      checkpoint_path = tf.train.latest_checkpoint(checkpoints_dir_y)
    else:
      checkpoint_path = checkpoints_dir_y

    eval_dir = checkpoints_dir_y + '/eval'
    if FLAGS.use_slim_stream_eval:
        eval_dir += '_slim'

    tf.logging.info('Evaluating %s' % checkpoint_path)

    my_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint_path)

    ##========================
    tf.logging.info('*********** Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                                          time.gmtime()))
    if use_placeholders:
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(eval_dir, sess.graph)
            def eval_loop_pl(sess, name_, filenames_=None, labels_=None, threshold_=None):

                images_pl_ = images_pl
                probabilities_op_pl_ = probabilities_op

                results_file_name1 = eval_dir + '/' + name_ + '_pl'
                tag_name = 'Evaluation/%s/whole_set_accuracy_pl' % (name_)
                if FLAGS.oversample_at_eval:
                    results_file_name1 = results_file_name1 + '_oversample'
                    tag_name += '_oversample'
                results_file_name1 = results_file_name1 + '.txt'
                accuracyy, acc_at_in_thr, eer_thr, eer_thr_max = classifier_pl.evaluate_loop_placeholder(sess, probabilities_op_pl_, images_pl_,
                                                       filenames_, categories_file, labels_,
                                                       results_file=results_file_name1, threshold=threshold_,
                                                       batch_size=eval_batch_size,
                                                       summary_op=my_summary_op, summary_writer=writer)

                summary_str = tf.Summary()
                summary_str.value.add(tag=tag_name, simple_value=accuracyy)
                if threshold_ is not None:
                    if isinstance(threshold_, list):
                        summary_str.value.add(tag=tag_name + '/atGivenThr', simple_value=max(acc_at_in_thr))
                    else:
                        summary_str.value.add(tag=tag_name + '/atGivenThr', simple_value=acc_at_in_thr)
                writer.add_summary(summary_str, sess.run(tf_global_step))
                if threshold_ is None:
                    tf.logging.info('-----------  %s: [%.4f]', tag_name, accuracyy)
                else:
                    if isinstance(threshold_, list):
                        tf.logging.info('-----------  %s: [%.4f]. At given threshold (%s): [%s]', tag_name,
                                        accuracyy, ','.join(str(thr) for thr in threshold_),
                                        ','.join(str(acc) for acc in acc_at_in_thr))
                    else:
                        tf.logging.info('-----------  %s: [%.4f]. At given threshold (%.4f): [%.4f]', tag_name, accuracyy,
                                        threshold_, acc_at_in_thr)

                if eer_thr_max > 0:
                    eer_thr = [eer_thr, eer_thr_max]
                return eer_thr

            # =============
            restore_fn(sess)
            eer_thr = eval_loop_pl(sess, name_=FLAGS.dataset_split_name_y,
                         filenames_=filenames, labels_=labels)
            if FLAGS.dataset_split_name_y2 is not None:
                eval_loop_pl(sess, name_=FLAGS.dataset_split_name_y2,
                             filenames_=filenames_2, labels_=labels_2, threshold_=eer_thr)

    else:
        ## ===========================
        def eval_loop(sess, eval_ops_, num_batches_, name_, accuracy_value_=None, probabilities_op_=None,
                      eval_ops_slim_=None, threshold_=None, filenames_op_=None):

            if FLAGS.use_slim_stream_eval:
                evaluation_y.evaluate_loop_slim_streaming_metrics(sess, num_batches_, eval_ops_)
                tf.logging.info('-----------  %s: Final Streaming Accuracy[%s]: %.4f',
                                time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()),
                                name_, sess.run(accuracy_value_) * 100)
                eer_thr = None
            else:
                if probabilities_op_ is not None:
                    eval_ops_.append(probabilities_op_)
                results_file_name = eval_dir + '/incorrect_filenames_' + name_ + '.txt'
                accuracyy, _, _, _, _, acc_at_in_thr, eer_thr, eer_thr_max = evaluation_y.evaluate_loop_y(sess, num_batches_,
                                                                                                          eval_batch_size,
                                                                                                          eval_ops_,
                                                                                                          eval_ops_slim=eval_ops_slim_,
                                                                                                          threshold=threshold_,
                                                                                                          filenames=filenames_op_,
                                                                                                          results_file=results_file_name)
                if accuracy_value_ is not None:
                    tf.logging.info('-----------  %s: Final Streaming Accuracy[%s]: %.4f',
                                    time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()),
                                    name_, sess.run(accuracy_value_) * 100)
                tag_name = 'Evaluation/%s/whole_set_accuracy' % (name_)
                summary_str = tf.Summary()
                summary_str.value.add(tag=tag_name, simple_value=accuracyy)
                if threshold_ is not None:
                    if isinstance(threshold_, list):
                        summary_str.value.add(tag=tag_name + '/atGivenThr', simple_value=max(acc_at_in_thr))
                    else:
                        summary_str.value.add(tag=tag_name + '/atGivenThr', simple_value=acc_at_in_thr)
                sv.summary_writer.add_summary(summary_str, sess.run(tf_global_step))
                if threshold_ is None:
                    tf.logging.info('-----------  %s: [%.4f]', tag_name, accuracyy)
                else:
                    if isinstance(threshold_, list):
                        tf.logging.info('-----------  %s: [%.4f]. At given threshold (%s): [%s]', tag_name,
                                        accuracyy, ','.join(str(thr) for thr in threshold_),
                                        ','.join(str(acc) for acc in acc_at_in_thr))
                    else:
                        tf.logging.info('-----------  %s: [%.4f]. At given threshold (%.4f): [%.4f]', tag_name, accuracyy,
                                        threshold_, acc_at_in_thr)

            if eer_thr_max>0:
                eer_thr = [eer_thr, eer_thr_max]
            return eer_thr

        sv = tf.train.Supervisor(logdir=eval_dir, summary_op=None, saver=None, init_fn=restore_fn)
        with sv.managed_session() as sess:
            #######################################
            if FLAGS.use_slim_stream_eval:
                eval_ops = eval_ops_slim
                eval_ops_2 = eval_ops_slim_2
            else:
                eval_ops = eval_ops_y
                eval_ops_2 = eval_ops_y_2
            eer_threshold = eval_loop(sess, eval_ops, num_batches, name_=FLAGS.dataset_split_name_y,
                                      eval_ops_slim_=eval_ops_slim, accuracy_value_=accuracy_value,
                                      probabilities_op_=probabilities_op, filenames_op_=filenames_op)
            if FLAGS.dataset_split_name_y2 is not None:
                eval_loop(sess, eval_ops_2, num_batches_2, name_=FLAGS.dataset_split_name_y2,
                          eval_ops_slim_=eval_ops_slim_2, accuracy_value_=accuracy_value_2,
                          probabilities_op_=probabilities_op_2, filenames_op_=filenames_op_2,
                          threshold_=eer_threshold)

            summaries = sess.run(my_summary_op)
            sv.summary_computed(sess, summaries)


    tf.logging.info('********* Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                                                      time.gmtime()))

if __name__ == '__main__':
  tf.app.run()

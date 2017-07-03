import csv
import numpy as np
import os
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from common import metrics


#############################################################################
def report_accuracy(pred_scores_all, labels_all, threshold, display_details=False):

    if isinstance(threshold, list):
        acc_at_in_thr = []
        for threshold_i in threshold:
            n_all = len(labels_all)
            acc1, far1, frr1, hter1 = metrics.Accuracy(pred_scores_all, labels_all, threshold_i)
            corr = round(acc1 * n_all)
            incorrect = n_all - corr
            logging.info(
                "----------- Accuracy at given threshold [%.4f](%%); %.2f%%, FAR: %.4f , FRR: %.4f, HTER: %.4f%%. N_incorrect = %d/%d",
                threshold_i, acc1 * 100, far1, frr1, hter1, incorrect, n_all)
            acc_at_in_thr.extend([acc1 * 100])

    else:
        n_all = len(labels_all)
        acc1, far1, frr1, hter1 = metrics.Accuracy(pred_scores_all, labels_all, threshold)
        corr = round(acc1 * n_all)
        incorrect = n_all - corr
        logging.info(
            "----------- Accuracy at given threshold [%.4f](%%); %.2f%%, FAR: %.4f , FRR: %.4f, HTER: %.4f%%. N_incorrect = %d/%d",
            threshold, acc1 * 100, far1, frr1, hter1, incorrect, n_all)
        acc_at_in_thr = acc1 * 100


    eer_thr, mer_thr, eer_thr_max = metrics.roc_det(pred_scores_all, labels_all, display_details=display_details)
    acc2, far2, frr2, hter2 = metrics.Accuracy(pred_scores_all, labels_all, eer_thr)
    corr = round(acc2 * n_all)
    incorrect = n_all - corr
    logging.info(
        "----------- Accuracy at eer threshold [%.4f](%%); %.2f%%, FAR: %.4f , FRR: %.4f, HTER: %.4f%%. N_incorrect = %d/%d",
        eer_thr, acc2 * 100, far2, frr2, hter2, incorrect, n_all)

    return acc_at_in_thr, eer_thr, eer_thr_max


#############################################################################
def write_results_file(results_file, filenames_all, labels_all, pred_scores_all, ind_incorrect, categories_list=None):
    if os.path.exists(results_file):
        os.rename(results_file, results_file+'-old.txt')

    write_scores=False
    if len(pred_scores_all)>0:
        write_scores=True
    incorrect_filenames_all2 = [filenames_all[j] for j in ind_incorrect]
    incorrect_labels_all2 = [labels_all[j] for j in ind_incorrect]
    pos_pred_scores_for_incorrect = [pred_scores_all[j] for j in ind_incorrect]

    print('Creating output file %s' % results_file)
    with open(results_file, 'w') as lFile:
        writer = csv.writer(lFile)
        writer.writerow(('filename', 'true_label', 'pos_pred_score'))
        for i, filename in enumerate(incorrect_filenames_all2):
            true_label = incorrect_labels_all2[i]
            if categories_list is not None:
                if write_scores:
                    row=(incorrect_filenames_all2[i], '%d[%s]' % (true_label, categories_list[true_label]), '%.4f' % pos_pred_scores_for_incorrect[i])
                else:
                    row = (incorrect_filenames_all2[i], '%d[%s]' % (true_label, categories_list[true_label]))
            else:
                if write_scores:
                    row = (incorrect_filenames_all2[i], true_label, '%.4f' % pos_pred_scores_for_incorrect[i])
                else:
                    row = (incorrect_filenames_all2[i], true_label)

            writer.writerow(row)

########################################################################3
def get_eval_ops(logits, labels, one_hot=False, scope='', calc_accuracy=True):
    """Evaluate the quality of the logits at predicting the label.
      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, NUM_CLASSES).
      Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
      """
    print('Evaluation Ops..')
    with tf.name_scope(scope):
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label's is was in the top k (here k=1)
        # of all logits for that example.
        # labels = tf.cast(labels, tf.int64)
        if one_hot:
            labels = tf.argmax(labels, 1)
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        num_correct = tf.reduce_sum(tf.cast(top_1_op, tf.float32))

        if calc_accuracy:
            acc_percent = tf.divide(num_correct, labels.shape[0].value)
        else:
            acc_percent = tf.constant(0.0)

        # =============
        y_const = tf.constant(-1, dtype=labels.dtype)
        y_greater = tf.greater(labels, y_const)
        n_all = tf.reduce_sum(tf.cast(y_greater, tf.float32))

        return top_1_op, acc_percent * 100.0, num_correct, n_all, labels


########################################################################
def get_eval_ops_slim(logits, labels, one_hot=False, scope=''):
    slim = tf.contrib.slim
    with tf.name_scope(scope + '/Streaming'):
        if one_hot:
            labels = tf.argmax(labels, 1)

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        })
        return names_to_values, names_to_updates


########################################################################
def evaluate_loop_slim_streaming_metrics(sess, num_val_iter, eval_ops):
    """ They have no return,s each metrisc write to summary .. so just monitor the tensorboard"""

    step = 0
    while step < num_val_iter:
        sess.run(eval_ops)
        logging.info('Evaluation [%d/%d]', step+1, num_val_iter)
        step += 1


########################################################################
def evaluate_loop_y(sess, num_val_iter, batch_size, eval_ops, display_details=False, threshold=None, filenames=None,
                    results_file=None, eval_ops_slim=None):

    total_sample_count = num_val_iter * batch_size
    logging.info('%s: starting evaluation on Examples: %d, for %d iter.' % (
        datetime.now(), total_sample_count, num_val_iter))

    # Counts the number of correct predictions.
    count_top_1 = 0.0
    n_correct_all = 0
    step = 0
    n_all = 0
    pred_scores_all = []
    labels_all = []
    filenames_all = []
    ind_incorrect = []
    incorrect_filenames_all = []
    incorrect_filenames_correct_labels = []

    start_time = time.time()
    while step < num_val_iter:
        if filenames is None:
            val_results = sess.run(eval_ops)  # => [top_1_op, acc, n_correct, n_in_batch, labels_in_batch ]
        else:
            if eval_ops_slim is None:
                val_results, filenames_in_batch = sess.run([eval_ops, filenames])
            else:
                val_results, filenames_in_batch, _ = sess.run([eval_ops, filenames, eval_ops_slim])
            filenames_all.extend(filenames_in_batch)
        top_1 = val_results[0]
        acc = val_results[1]
        n_correct = val_results[2]
        n_in_batch = val_results[3]
        labels_in_batch = val_results[4]
        labels_all.extend(labels_in_batch)
        if filenames is not None:
            ind_incorrect_tmp = np.where(top_1 == False)[0]
            incorrect_filenames_all.extend(list(filenames_in_batch[ind_incorrect_tmp]))
            incorrect_filenames_correct_labels.extend(list(labels_in_batch[ind_incorrect_tmp]))
            ind_incorrect.extend(list(ind_incorrect_tmp + (step * batch_size)))
        if len(val_results)>5:
            probabilities = val_results[5]
            pred_scores = probabilities[:, 1]  # assume score to be the pred prob of the positive class [1]
            pred_scores_all.extend(pred_scores)

        count_top_1 += np.sum(top_1)
        n_correct_all += n_correct
        n_all += n_in_batch
        step += 1
        logging.info('Evaluation [%d/%d]. n_correct/n_in_batch: %d/%d', step, num_val_iter, n_correct, n_in_batch)

    total_sample_count = n_all
    accuracyy = n_correct_all * 100.0 / total_sample_count

    eval_time = time.time() - start_time
    logging.info('--- Evaluated on Examples: %d, for %d iter.' % (total_sample_count, num_val_iter))
    logging.info('--- Testing time: %.3f' % (eval_time))

    acc_percent=0.0
    precision_at_1=0.0

    if filenames is not None:
        print ('------ Incorrect_filenames based on top_1 (0.5 threshold):')
        print (incorrect_filenames_all)
        write_results_file(results_file, filenames_all, labels_all, pred_scores_all, ind_incorrect)

    acc_at_in_thr, eer_thr, eer_thr_max = 0, None, None
    if len(val_results) > 5:
        if threshold is None:
            threshold = 0.5
        else:
            report_accuracy(pred_scores_all, labels_all,0.5, display_details)
        acc_at_in_thr, eer_thr, eer_thr_max = report_accuracy(pred_scores_all, labels_all, threshold, display_details)


    return accuracyy, n_correct_all, acc_percent, precision_at_1, total_sample_count, acc_at_in_thr, eer_thr, eer_thr_max


############################################################
from data import helpers
from pipeline_tf import image_utils
from preprocessing import preprocessing_factory

class Classifier_PL():
    def __init__(self, eval_image_size, n_channels, preprocessing_name, encode_type='JPEG',
                 oversample=False, per_image_standardization=False,
                 vgg_sub_mean_pixel=None, vgg_resize_side_in=None,
                 vgg_use_aspect_preserving_resize=None):

        self.tensor1_pl = tf.placeholder(tf.float32, (None, eval_image_size, eval_image_size, n_channels))
        self.tensor2_pl = tf.placeholder(tf.float32, (None, eval_image_size, eval_image_size, n_channels))
        self.tf_concat_op = tf.concat([self.tensor1_pl, self.tensor2_pl], 0)

        image_preprocessing_fn_at_eval = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)
        image_crop_processing_fn = preprocessing_factory.get_img_crop_preprocessing(preprocessing_name)
        self.raw_images_pl = tf.placeholder(tf.float32, [None, None, 3])
        self.preprocessed_img_crops = image_utils.get_preprocessed_img_crops(self.raw_images_pl, preprocessing_name,
                                                                             eval_image_size, eval_image_size,
                                                                             image_preprocessing_fn_at_eval, image_crop_processing_fn,
                                                                             oversample=oversample,
                                                                             per_image_standardization=per_image_standardization,
                                                                             vgg_sub_mean_pixel=vgg_sub_mean_pixel,
                                                                             vgg_resize_side_in=vgg_resize_side_in,
                                                                             vgg_use_aspect_preserving_resize=vgg_use_aspect_preserving_resize)
        self.coder = image_utils.ImageCoder_TF()
        self.encode_type = encode_type

    ############################################################
    def classify_batch(self, sess, softmax_output, images_pl, image_files,
                       summary_op=None, summary_writer=None):
        first = True
        batch_sizes = []
        # crops_all=[]
        for image_file in image_files:
            _, im_batch = image_utils.make_batch_from_img_pl(sess, self.preprocessed_img_crops, self.raw_images_pl, image_file,
                                                             self.coder, encode_type=self.encode_type,
                                                             summary_op=summary_op, summary_writer=summary_writer)
            batch_sz = im_batch.shape[0]
            batch_sizes.extend([batch_sz])
            if first:
                y_batch_ = im_batch
                first = False
            else:
                y_batch_ = sess.run(self.tf_concat_op, feed_dict={self.tensor1_pl: y_batch_, self.tensor2_pl: im_batch})

        batch_results_all = sess.run(softmax_output, feed_dict={images_pl: y_batch_})
        n = 0
        img = 0
        output_all = np.zeros([len(batch_sizes), 2])
        for sz in batch_sizes:
            batch_results = batch_results_all[n:n + sz]
            n += sz  # .value
            output = batch_results[0]
            batch_sz = batch_results.shape[0]
            for i in range(1, batch_sz):
                output = output + batch_results[i]

            output /= batch_sz
            output_all[img] = output
            img += 1

        return output_all

    ##########################################################################
    def classify(self, sess, softmax_output, images_pl, image_file):
        # print('Running file %s' % image_file)
        crops, im_batch = image_utils.make_batch_from_img_pl(sess, self.preprocessed_img_crops, self.raw_images_pl, image_file,
                                                             self.coder, encode_type=self.encode_type)

        batch_results = sess.run(softmax_output, feed_dict={images_pl: im_batch})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
        logging.info('   - Ran batch of %d images' % batch_sz)
        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz

        return output

    ############################################################
    def evaluate_loop_placeholder(self, sess, probabilities_op, images_pl,
                                  filenames, categories_file, labels, threshold=None,
                                  results_file=None, batch_size=1, summary_op=None, summary_writer=None):

        output = None
        writer=None
        if results_file is not None:
            if os.path.exists(results_file):
                os.rename(results_file, results_file + '-old.txt')
            print('Creating output file %s' % results_file)
            output = open(results_file, 'w')
            writer = csv.writer(output)
            writer.writerow(('file', 'predicted_label', 'score_of_prediction'))

        label_list, _ = helpers.get_lines_in_file(categories_file, read=True)
        n = len(filenames)
        pred_scores_all=np.zeros(n)
        ind_incorrect=[]

        i=0
        if batch_size > 1:
            final_batch = False
            while not final_batch:
                if (i+batch_size)>len(filenames):
                    image_files = filenames[i:len(filenames)]
                    final_batch=True
                else:
                    image_files = filenames[i:i+batch_size]

                if len(image_files)<1:
                    break

                logging.info(time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()) + '  Running files [%d-%d/%d]' % (i, i+batch_size-1, n))
                probabilities = self.classify_batch(sess, probabilities_op, images_pl,
                                         image_files, summary_op=summary_op, summary_writer=summary_writer)
                pred_scores_all[i:i+batch_size] = probabilities[:,1]  # assume score to be the pred prob of the positive class [1]
                best = np.argmax(probabilities, axis=1)
                for m, image_file in enumerate(image_files):
                    if not labels[i+m] == best[m]:
                        ind_incorrect.extend([i+m])
                    if writer is not None:
                        writer.writerow((image_file, label_list[best[m]], '%.2f' % probabilities[m,best[m]]))
                i += batch_size

        else: # batch_size = 1
            for i,image_file in enumerate(filenames):
                if image_file is None: continue
                # try:
                logging.info(time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()) + '  Running file [%d/%d]: %s' % (i,n, image_file))
                probabilities = self.classify(sess, probabilities_op, images_pl,image_file)
                pred_scores_all[i] = probabilities[1]  # assume score to be the pred prob of the positive class [1]
                best = np.argmax(probabilities)
                if not labels[i]==best:
                    ind_incorrect.extend([i])
                if writer is not None:
                    writer.writerow((image_file, label_list[best], '%.2f' % probabilities[best]))

        ###########
        if output is not None:
            output.close()

        acc, eer_thr, eer_thr_max = report_accuracy(pred_scores_all, labels, threshold=0.5)
        acc_at_in_thr = 0
        if threshold is not None:
            acc_at_in_thr, _ , _= report_accuracy(pred_scores_all, labels, threshold)

        if results_file is not None:
            new_results_file = os.path.dirname(results_file) + '/incorrect_filenames_' + os.path.basename(results_file)
            write_results_file(new_results_file, filenames, labels, pred_scores_all, ind_incorrect, label_list)

        return acc, acc_at_in_thr, eer_thr, eer_thr_max
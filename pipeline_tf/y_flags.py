import tensorflow as tf
from pipeline_tf import settings_default

## =================================================================
#################
# Training steps
#################

tf.app.flags.DEFINE_integer('max_number_of_steps', settings_default._MAX_TRAIN_STEPS,
                            """Number of batches to run. The maximum number of training steps.""")

_CHKPT_STEP = settings_default._CHKPT_STEP
_SUMMARY_STEP = settings_default._SUMMARY_STEP

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', settings_default._DISPLAY_STEP,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'validation_every_n_steps', settings_default._VAL_STEP,
    'The frequency with which training set is evaluated.')

tf.app.flags.DEFINE_integer(
    'test_every_n_steps', -1,
    'The frequency with which validation/test set is evaluated.')

## =================================================================
tf.app.flags.DEFINE_boolean('start_labels_at_1', False, ##True,
                            """Whether label 0 is reserved for background during building data tf records. let Default = False""")

## =================================================================
#######################
# Dataset Flags #
#######################
tf.app.flags.DEFINE_string(
    'dataset_name', settings_default.DATASET_NAME,
    'The name of the dataset to load.')

tf.app.flags.DEFINE_string('tfRecords_dir', None,
                           'Output data directory')
tf.app.flags.DEFINE_string('encode_type', settings_default.F_encode_type,
                           'PNG or JPEG')

## =================================================================
tf.app.flags.DEFINE_string(
    'model_name',  settings_default.MODEL,
    'The name of the architecture to train.')

tf.app.flags.DEFINE_integer('image_size', settings_default.IMG_SIZE,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('vgg_resize_side', None,
                            """.""")
tf.app.flags.DEFINE_boolean('vgg_use_aspect_preserving_resize', True,
                            """""")
tf.app.flags.DEFINE_boolean('vgg_sub_mean_pixel', True,
                            """""")
tf.app.flags.DEFINE_integer('channels', settings_default.nCHANNELS,
                            """Color or grayscale.""")
tf.app.flags.DEFINE_integer('batch_size', settings_default.BATCH_SIZE,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_boolean('per_image_standardization', settings_default.USE_PER_IM_STD,
                            """at image preprocessing during training/eval, sub mean and divide by std from img itself""")

tf.app.flags.DEFINE_string('checkpoints_dir_prefix', None,
                           """Directory where to write/read event logs """
                           """and checkpoint.""")

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_string(
    'optimizer', settings_default.OPTIMIZER,
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'weight_decay', settings_default.WEIGHT_DECAY,
    'The weight decay on the model weights.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_float('num_epochs_per_decay', settings_default.LR_DECAY_EPOCS,
                          """Epochs after which learning rate decays.""")

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('initial_learning_rate', settings_default.LR_INITIAL, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor',
    settings_default.LR_DECAY_FACTOR,
    'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', settings_default.MOVING_AVERAGE_DECAY,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

## =================================================================
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset during train.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
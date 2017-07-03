## ==================================================================
# ======== Data:
# DATASET_NAME= 'catsDogs'
# DATASET_NAME = 'ATVS'
DATASET_NAME = 'Warsaw'
# DATASET_NAME = 'MobBioFake'

# ========
nCHANNELS = 3
# nCHANNELS = 1 # GRAYSCALE

F_encode_type = 'PNG'
# F_encode_type = 'JPEG'

# ===============
USE_PER_IM_STD = False
# USE_PER_IM_STD = True

## =================================================================
# ======== Training:
MODEL = 'cifar10'  # 'cifar10', or 'inception',or  'alexnet', 'googlenet', 'overfeat', 'vgg', 'spoofnet_y'
# MODEL = 'alexnet'
MODEL = 'spoofnet_y'

# ===============
OPTIMIZER='rmsprop' # 'RMSPROP' (used by inception), 'GRADDESCENT' (used by cifar10 and other tutorials), 'ADAGRAD' (AdaptiveGrad), 'ADADELTA', 'ADAM', 'MOM' (nesterov momentum),
# OPTIMIZER = 'sgd'
#     'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
#     '"ftrl", "momentum", "sgd" or "rmsprop".')

WEIGHT_DECAY = 0.0004

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

# ===============
LR_INITIAL = 0.01 ## YY: def for inception = 0.1
LR_DECAY_EPOCS = 200 #60 ## YY: def for inception = 30
LR_DECAY_FACTOR = 0.1 ## YY: 'gamme' in caffe ... ## YY: def for inception = 0.16

# ===============
_VAL_STEP = 50
_MAX_TRAIN_STEPS = 2500
_CHKPT_STEP = 50 #500  # 5000
_DISPLAY_STEP = 10
_SUMMARY_STEP = 10

## =================================================================
IMG_SIZE = None
BATCH_SIZE = 128
if (MODEL == 'cifar10'):
    IMG_SIZE = 24
    IMG_SIZE = 112
    BATCH_SIZE = 128
elif MODEL == 'spoofnet_y':
    IMG_SIZE = 24
    IMG_SIZE = 112
    BATCH_SIZE = 128
elif MODEL == 'alexnet':
    IMG_SIZE = 227
    BATCH_SIZE = 64 #32 #64
    _VAL_STEP  = 40
    _CHKPT_STEP = 20
elif MODEL == 'inception':
    IMG_SIZE = 299
    BATCH_SIZE = 32
elif MODEL == 'googlenet':
    pass
elif MODEL == 'overfeat':
    pass
elif MODEL == 'vgg':
    pass

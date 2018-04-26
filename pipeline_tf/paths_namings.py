

## =================================================================
## =================================================================
import os.path
def generate_name_suffix(GRAYSCALE, F_encode_type):
    name_suffix = 'TFCoder'
    if GRAYSCALE:
        name_suffix += '-grey'
    name_suffix += '-%s' % (F_encode_type)

    return name_suffix


## =================================================================
def generate_checkpoints_dir(FLAGS_Y, image_size):

    if FLAGS_Y.channels == 3:
        set_grayscale = False
    elif FLAGS_Y.channels == 1:
        set_grayscale = True

    name_suffix = generate_name_suffix(set_grayscale, FLAGS_Y.encode_type)

    # ===========================================
    checkpoints_dir_prefix, _ = get_dataset_chkpoints_dir_pre(FLAGS_Y.dataset_name)
    print ( ' ----------- ', checkpoints_dir_prefix)
    MODEL = FLAGS_Y.model_name
    OPTIMIZER = FLAGS_Y.optimizer

    IMG_SIZE = image_size
    LR_INITIAL = FLAGS_Y.initial_learning_rate
    WD=FLAGS_Y.weight_decay
    BATCH_SIZE = FLAGS_Y.batch_size
    USE_PER_IM_STD = FLAGS_Y.per_image_standardization

    # ===========================================
    sub_dir = os.path.join(name_suffix, MODEL, OPTIMIZER)
    train_postfix = 'I' + str(IMG_SIZE) + '_iniLR' + str(LR_INITIAL) + '_B' + str(BATCH_SIZE)

    if USE_PER_IM_STD:
        train_postfix += '_perImStd'

    if WD is not None:
        train_postfix += '_wd' + str(WD)

    checkpoints_dir = os.path.join(checkpoints_dir_prefix, sub_dir, train_postfix)
    print(checkpoints_dir)

    return checkpoints_dir


## =================================================================
def file_pattern_tfrecords(FLAGS_Y, tfRecords_dir, split_name):
    if FLAGS_Y.channels == 3:
        set_grayscale = False
    elif FLAGS_Y.channels == 1:
        set_grayscale = True

    name_suffix = generate_name_suffix(set_grayscale, FLAGS_Y.encode_type)
    print(name_suffix)
    tf_record_pattern = os.path.join(tfRecords_dir, '%s-%s*' % (split_name, name_suffix))

    return tf_record_pattern


## =================================================================
from data.datasets_paths import TF_CHECKPOINTS_ROOT_DIR, TFRecords_ROOT_DIR, datasets_paths_map
def get_dataset_chkpoints_dir_pre(dataset_name):

    # ===================================
    if dataset_name not in datasets_paths_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    path_dataset = datasets_paths_map[dataset_name]()
    checkpoints_dir = os.path.join(TF_CHECKPOINTS_ROOT_DIR , path_dataset.get_name())
    return checkpoints_dir, path_dataset


# ===========================
def get_dataset_paths_and_settings(dataset_name):
    checkpoints_dir_prefix, path_dataset = get_dataset_chkpoints_dir_pre(dataset_name)

    labels_file_name = path_dataset.categories_file()
    validation_exists = False
    regenerate_label_files = False

    TRAIN_IMGS_DIR = None
    VAL_IMGS_DIR = None
    TEST_IMGS_DIR = None

    TRAIN_CSV_FILE = None
    VAL_CSV_FILE = None
    TEST_CSV_FILE = None

    # ===================================
    if dataset_name == 'catsDogs':
        TFRecords_DIR = TFRecords_ROOT_DIR + "/" + path_dataset.get_name()

        TRAIN_IMGS_DIR = path_dataset.sub_dir('train')
        VAL_IMGS_DIR = None
        TEST_IMGS_DIR = path_dataset.sub_dir('test')

        TRAIN_CSV_FILE = path_dataset.csv_file('train')
        VAL_CSV_FILE = path_dataset.csv_file('validation')
        TEST_CSV_FILE = path_dataset.csv_file('test')

    elif (dataset_name == 'Warsaw') or (dataset_name == 'ATVS') or (dataset_name == 'MobBioFake'):
        TFRecords_DIR = TFRecords_ROOT_DIR + "/" + path_dataset.get_name()

        TRAIN_IMGS_DIR = path_dataset.root_dir()
        VAL_IMGS_DIR = path_dataset.root_dir()

        TRAIN_CSV_FILE = path_dataset.csv_file('train')
        VAL_CSV_FILE = path_dataset.csv_file('validation')

        labels_file_name = path_dataset.categories_file()

    ################===========================
    imgs_sub_dirs = {'train': TRAIN_IMGS_DIR,
                'validation': VAL_IMGS_DIR,
                'test': TEST_IMGS_DIR}

    csv_files = {'train': TRAIN_CSV_FILE,
                     'validation': VAL_CSV_FILE,
                     'test': TEST_CSV_FILE}

    return checkpoints_dir_prefix, TFRecords_DIR, imgs_sub_dirs, csv_files, labels_file_name

import os.path

# ROOT_DIR = "/media/yomna/Work/Y_Work"
ROOT_DIR = "D:\\"
# ==========
DBs_DIR = os.path.join(ROOT_DIR , "0_ImgsDB")

IRIS_ROOT_DIR = os.path.join(DBs_DIR , "Spoof_DBs","IRIS")
WARSAW_SUB_DIR = 'LivDet-Iris-2013-Warsaw'
ATVS_SUB_DIR = 'BioSec_ATVS'
MOBBIOFAKE_SUB_DIR = "MobBioFake"

CATS_DOGS_DIR = os.path.join(DBs_DIR , "Kaggle_CatsVsDogs")

TF_CHECKPOINTS_ROOT_DIR = os.path.join(ROOT_DIR , 'data','models-training_snapshots_tensorflow')
TFRecords_ROOT_DIR = os.path.join(ROOT_DIR , 'data','data_tf')

####################################################################
from abc import ABCMeta, abstractmethod
class DatasetPaths_Y(object):
    __metaclass__ = ABCMeta
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    @abstractmethod
    def root_dir(self):
        return

    def sub_dir(self, subset):
        return

    @abstractmethod
    def csv_files_dir(self):
        return

    @abstractmethod
    def csv_file(self, subset):
        return

    @abstractmethod
    def categories_file(self):
        return

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']


###############################################################
class IrisATVS_Paths(DatasetPaths_Y):
    def __init__(self):
        super(IrisATVS_Paths, self).__init__('biosec')

    def root_dir(self):
        return os.path.join(IRIS_ROOT_DIR , ATVS_SUB_DIR , "/ATVS-FIr_DB")

    def csv_files_dir(self):
        return os.path.join(IRIS_ROOT_DIR , ATVS_SUB_DIR , "/labelFiles")

    def csv_file(self, subset):
        assert subset in self.available_subsets(), self.available_subsets()
        if subset == 'train':
            return os.path.join(self.csv_files_dir() ,self._name + "_spoof_train.txt")

        if subset == 'validation':
            return os.path.join(self.csv_files_dir() , + self._name + "_spoof_test.txt")

    def categories_file(self):
        return os.path.join(self.csv_files_dir(), + self._name + "_categories.txt")


###############################################################
class IrisWarsaw_Paths(DatasetPaths_Y):
    def __init__(self):
        super(IrisWarsaw_Paths, self).__init__('warsaw')

    def root_dir(self):
        return os.path.join(IRIS_ROOT_DIR, WARSAW_SUB_DIR ,"PNG")

    def csv_files_dir(self):
        return os.path.join(IRIS_ROOT_DIR , WARSAW_SUB_DIR , "labelFiles")

    def csv_file(self, subset):
        assert subset in self.available_subsets(), self.available_subsets()
        if subset == 'train':
            return os.path.join(self.csv_files_dir() , self._name + "_spoof_train.txt")

        if subset == 'validation':
            return os.path.join(self.csv_files_dir(), self._name + "_spoof_test.txt")

    def categories_file(self):
        return os.path.join(self.csv_files_dir() , self._name + "_categories.txt")


###############################################################
class IrisMobbiofake_Paths(DatasetPaths_Y):
    def __init__(self):
        super(IrisMobbiofake_Paths, self).__init__('mobbiofake')

    def root_dir(self):
        return IRIS_ROOT_DIR + MOBBIOFAKE_SUB_DIR

    def csv_files_dir(self):
        return self.root_dir() + "/labelFiles"

    def csv_file(self, subset):
        assert subset in self.available_subsets(), self.available_subsets()
        if subset == 'train':
            return self.csv_files_dir() + '/' + self._name + "_spoof_train.txt"

        if subset == 'validation':
            return self.csv_files_dir() + '/' + self._name + "_spoof_test.txt"

    def categories_file(self):
        return self.csv_files_dir() + '/' + self._name + "_categories.txt"

###############################################################
class CatsDogs_Paths(DatasetPaths_Y):
    def __init__(self):
        super(CatsDogs_Paths, self).__init__('catsDogs')

    def root_dir(self):
        return CATS_DOGS_DIR

    def sub_dir(self, subset):
        assert subset in self.available_subsets(), self.available_subsets()
        if subset == 'train':
            return self.root_dir() + '/train'
        if subset == 'validation':
            return None
        if subset == 'test':
            return self.root_dir() + '/test1'

    def csv_files_dir(self):
        return self.root_dir()

    def csv_file(self, subset):
        assert subset in self.available_subsets(), self.available_subsets()
        if subset == 'train':
            return self.csv_files_dir() + '/train.txt'
        if subset == 'validation':
            return self.csv_files_dir() + '/val.txt'
        if subset == 'test':
            return self.csv_files_dir() + '/test.txt'

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation', 'test']

    def categories_file(self):
        return self.csv_files_dir() + '/labels.txt'


###############################################################
datasets_paths_map = {
    'catsDogs': CatsDogs_Paths,
    'Warsaw': IrisWarsaw_Paths,
    'ATVS': IrisATVS_Paths,
    'MobBioFake': IrisMobbiofake_Paths,
}


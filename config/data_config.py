import os

from easydict import EasyDict  # type: ignore

from utils.enums import PreprocessingType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_config = EasyDict()

# path to the directory with dataset files
data_config.path_to_data = os.path.join(ROOT_DIR, 'data', 'diamonds')
data_config.type = {
    'train': 'diamonds_train.csv',  # training set
    'valid': 'diamonds_validation.csv',  # valid set
    'test': 'diamonds_test.csv',  # test set
}

data_config.preprocess_type = PreprocessingType.normalization

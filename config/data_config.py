import os

from utils.enums import PreprocessingType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class Config(object):
    PATH_TO_DATA = os.path.join(ROOT_DIR, 'data', 'diamonds')
    TYPE = {
        'train': 'diamonds_train.csv',
        'valid': 'diamonds_validation.csv',
        'test': 'diamonds_test.csv',
    }
    PREPROCESS_TYPE = PreprocessingType.NORMALIZATION

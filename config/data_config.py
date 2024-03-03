import os

from utils.enums import PreprocessingType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class Config(object):
    PATH_TO_DATA = os.path.join(ROOT_DIR, 'dataset')
    TYPE = {
        'TRAIN': 'diamonds_train.csv',
        'VALID': 'diamonds_validation.csv',
        'TEST': 'diamonds_test.csv',
    }
    PREPROCESS_TYPE = PreprocessingType.NORMALIZATION

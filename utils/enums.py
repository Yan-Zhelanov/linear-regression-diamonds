from enum import Enum


class SetType(Enum):
    """Data set type."""

    TRAIN = 1
    VALID = 2
    TEST = 3


class PreprocessingType(Enum):
    """Preprocessing type enum."""

    NORMALIZATION = 1
    STANDARTIZATION = 2

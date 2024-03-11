import numpy as np

from utils.enums import PreprocessingType

IntOrFloat = int | float

LOWER_BOUND = -1
UPPER_BOUND = 1


class Preprocessing(object):
    """A class for data preprocessing."""

    def __init__(self, preprocess_type: PreprocessingType):
        self._preprocess_type = preprocess_type
        self._min: IntOrFloat = 0
        self._max: IntOrFloat = 0
        self._mean: IntOrFloat = 0
        self._std: IntOrFloat = 0

    def fit(self, features: np.ndarray) -> None:
        """Initialize preprocessing function on training data.

        Args:
            features: feature array.
        """
        self._mean = np.mean(features)
        self._std = np.std(features)
        self._min = np.min(features)
        self._max = np.max(features)

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """Preprocesses features.

        Args:
            features: feature array.

        Returns:
            np.ndarray: preprocessed features.
        """
        if self._preprocess_type is PreprocessingType.NORMALIZATION:
            return self._normalization(features)
        return self._standardization(features)

    def _normalization(self, features: np.ndarray) -> np.ndarray:
        """Transform x by scaling each feature to a range [-1, 1].

        Using self.params['min'] and self.params['max'].

        Args:
            features: feature array.

        Returns:
            np.ndarray: normilized features.
        """
        return LOWER_BOUND + (UPPER_BOUND - LOWER_BOUND) * (
            features - self._min
        ) / (self._max - self._min)

    def _standardization(self, features: np.ndarray) -> np.ndarray:
        """Standardize x with self.params['mean'] and self.params['std']

        Args:
            features: feature array

        Returns:
            np.ndarray: standardized features.
        """
        return (features - self._mean) / self._std

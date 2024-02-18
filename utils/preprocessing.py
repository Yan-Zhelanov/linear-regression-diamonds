import numpy as np

from utils.enums import PreprocessingType


class Preprocessing(object):
    """A class for data preprocessing."""

    def __init__(self, preprocess_type: PreprocessingType):
        self.preprocess_type = preprocess_type

        # A dictionary with the following keys and values:
        #    - {'min': min values, 'max': max values} when preprocess_type is PreprocessingType.normalization
        #    - {'mean': mean values, 'std': std values} when preprocess_type is PreprocessingType.standardization
        self.params = None

        # Select the preprocess function according to self.preprocess_type
        self.preprocess_func = getattr(self, self.preprocess_type.name)

    def normalization(
        self, features: np.ndarray, init: bool = False,
    ) -> np.ndarray:
        """Transform x by scaling each feature to a range [-1, 1].

        Using self.params['min'] and self.params['max'].

        Args:
            features: feature array.
            init: initialization flag.

        Returns:
            np.ndarray: normilized features.
        """
        if init:
            # TODO: calculate min and max for each column in x with np.min, np.max
            #       store the values in self.params['min'] and self.params['max']
            pass

        # TODO: implement data normalization
        #       normalized_x = a + (b - a) * (x - self.params['min']) / (self.params['max'] - self.params['min']),
        #       where a = -1, b = 1
        raise NotImplementedError

    def standardization(
        self, features: np.ndarray, init: bool = False,
    ) -> np.ndarray:
        """Standardize x with self.params['mean'] and self.params['std']

        Args:
            features: feature array
            init: initialization flag

        Returns:
            np.ndarray: standardized features.
        """
        if init:
            # TODO: calculate mean and std for each column in x with np.mean, np.std
            #       store the values in self.params['mean'] and self.params['std']
            pass

        # TODO: implement data standardization
        #       standardized_x = (x - self.params['mean']) / self.params['std']
        raise NotImplementedError

    def train(self, features: np.ndarray) -> np.ndarray:
        """Initialize preprocessing function on training data."""
        return self.preprocess_func(features, init=True)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Return preprocessed data."""
        if self.params is None:
            raise ValueError(
                f'{self.preprocess_type.name} instance is not trained yet.'
                + "Please call 'train' first.",
            )
        return self.preprocess_func(features, init=False)

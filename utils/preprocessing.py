import numpy as np
from utils.enums import PreprocessingType


class Preprocessing:
    """A class for data preprocessing."""

    def __init__(self, preprocess_type: PreprocessingType):
        self.preprocess_type = preprocess_type

        # A dictionary with the following keys and values:
        #    - {'min': min values, 'max': max values} when preprocess_type is PreprocessingType.normalization
        #    - {'mean': mean values, 'std': std values} when preprocess_type is PreprocessingType.standardization
        self.params = None

        # Select the preprocess function according to self.preprocess_type
        self.preprocess_func = getattr(self, self.preprocess_type.name)

    def normalization(self, x: np.ndarray, init=False):
        """Transforms x by scaling each feature to a range [-1, 1] with self.params['min'] and self.params['max']

        Args:
            x: feature array
            init: initialization flag

        Returns:
            normalized_x (numpy.array)
        """
        if init:
            # TODO: calculate min and max for each column in x with np.min, np.max
            #       store the values in self.params['min'] and self.params['max']
            pass

        # TODO: implement data normalization
        #       normalized_x = a + (b - a) * (x - self.params['min']) / (self.params['max'] - self.params['min']),
        #       where a = -1, b = 1
        raise NotImplementedError

    def standardization(self, x: np.ndarray, init=False):
        """Standardizes x with self.params['mean'] and self.params['std']

        Args:
            x: feature array
            init: initialization flag

        Returns:
            standardized_x (numpy.array)
        """
        if init:
            # TODO: calculate mean and std for each column in x with np.mean, np.std
            #       store the values in self.params['mean'] and self.params['std']
            pass

        # TODO: implement data standardization
        #       standardized_x = (x - self.params['mean']) / self.params['std']
        raise NotImplementedError

    def train(self, x: np.ndarray):
        """Initializes preprocessing function on training data."""
        return self.preprocess_func(x, init=True)

    def __call__(self, x: np.ndarray):
        """Returns preprocessed data."""
        if self.params is None:
            raise Exception(f"{self.preprocess_type.name} instance is not trained yet. Please call 'train' first.")
        return self.preprocess_func(x, init=False)

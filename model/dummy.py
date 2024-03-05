import numpy as np


class Dummy(object):
    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self._mean = np.mean(target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self._mean)

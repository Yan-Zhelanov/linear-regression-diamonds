import numpy as np


def get_rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute RMSE.

    Args:
        targets: observed values
        predictions: predicted values

    Returns:
        float: computed RMSE value.
    """
    return np.sqrt(get_mse(targets, predictions))


def get_mse(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute MSE.

    Args:
        targets: observed values
        predictions: predicted values

    Returns:
        float: computed MSE value.
    """
    return np.mean((targets - predictions)**2)

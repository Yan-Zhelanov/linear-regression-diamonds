import numpy as np


def get_rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute RMSE.

    Args:
        targets: observed values
        predictions: predicted values

    Returns:
        float: computed RMSE value.
    """
    return np.sqrt(np.mean((targets - predictions)**2))

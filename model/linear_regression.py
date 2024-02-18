import pickle
import sys

import numpy as np


class LinearRegression:
    """A class for linear regression model implementation."""

    def __init__(self, number_bases: int, reg_coefficient: float = 0) -> None:
        """Initialize the linear regression model.

        Args:
            number_bases: the number of basis functions (or model weights
                vector shape in case of vanilla linear regression).
            reg_coefficient: regularization coefficient.
        """
        self._weights = np.random.randn(number_bases)
        self._reg_coefficient = reg_coefficient

    @staticmethod
    def _pseudo_inverse_matrix(
        matrix: np.ndarray, regularization: float | None = None,
    ) -> np.ndarray:
        """Compute the pseudo-inverse of a matrix using SVD.

        The pseudo-inverse (Φ^+) of the design matrix Φ can be computed using
        the formula: Φ^+ = V * Σ^+ * U^T
        where U, Σ, and V are the matrices resulting from the SVD of Φ.
        The Σ^+ is computed as:
        Σ'_{i,j} =
            | 1/Σ_{i,j}, if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
            | 0, otherwise
        and then:
            Σ^+ = Σ'^T
        where:
            - ε is the machine epsilon, which can be obtained in Python using:
                ε = sys.float_info.epsilon
            - N is the number of rows in the design matrix.
            - M is the number of base functions (without φ_0(x_i)=1).

        For regularization
        Σ'_{i,j} =
            | Σ_{i,j}/(Σ_{i,j}ˆ2 + λ) , if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
            | 0, otherwise

        Note that Σ'_[0,0] = 1/Σ_{i,j}
        """
        left_singular, singular, right_singular_transposed = np.linalg.svd(
            matrix,
        )
        pseudo_inverse_sigma = np.zeros((matrix.shape[1], matrix.shape[0]))
        epsilon = sys.float_info.epsilon
        threshold = epsilon * max(matrix.shape) * max(singular)
        mask = singular > threshold
        if regularization is None:
            pseudo_inverse_sigma[mask] = 1 / singular[mask]
        else:
            pseudo_inverse_sigma[mask] = singular[mask] / (
                singular[mask]**2 + regularization
            )
        return left_singular @ pseudo_inverse_sigma @ right_singular_transposed

    def _compute_weights(
        self, pseudo_inverse_plan_matrix: np.ndarray, targets: np.ndarray,
    ) -> None:
        """Computes the optimal weights using the normal equation.

        The weights (w) can be computed using the formula: w = Φ^+ * t
        where:
            - Φ^+ is the pseudoinverse of the design matrix and can be defined
                as Φ^+ = (Φ^T * Φ)^(-1) * Φ^T
            - t is the target vector.
        """
        pass

    def _compute_model_prediction(self, features: np.ndarray) -> np.ndarray:
        """Computes the predictions of the model.

        The prediction (y_pred) can be computed using the formula:
        y_pred = Φ * w^T
        where:
            - Φ is the design matrix.
            - w^T is the transpose of the weight vector.
        To compute multiplication in Python using numpy, you can use:
            - `numpy.dot(a, b)`
            OR
            - `a @ b`
        """
        pass

    def _compute_cost_function(self, plan_matrix, targets):
        """Computes the cost function value for the current weights.

        The cost function E(w) represents the mean squared error and is given
        by: E(w) = (1/N) * ∑(t - Φ * w^T)^2
        where:
            - Φ is the design matrix.
            - w is the weight vector.
            - t is the vector of target values.
            - N is the number of data points.

        For regularization:
            E(w) = (1/N) * ∑(t - Φ * w^T)^2 + λ * w^T * w

        TODO: Implement this method using numpy operations to compute the mean
            squared error. Do not use loops
        TODO: Add regularization
        """
        pass

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Trains the model using the normal equation."""
        pseudo_inverse_matrix = self._pseudo_inverse_matrix(inputs)

        # Training process
        self._compute_weights(pseudo_inverse_matrix, targets)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Returns model prediction."""
        predictions = self._compute_model_prediction(inputs)
        return predictions

    def save(self, filepath):
        """Saves trained model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self._weights, f)

    def load(self, filepath):
        """Loads trained model."""
        with open(filepath, 'rb') as f:
            self._weights = pickle.load(f)

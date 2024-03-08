from enum import Enum
from typing import Callable, Dict

import numpy as np


class BasisFunctionType(Enum):
    POLYNOMIAL = 1
    RBF = 2
    SIGMOID = 3
    FOURIER = 4
    WITHOUT = 5


class BasisFunctionTransform(object):
    """A class for transforming data features with basis functions.

    This class allows the transformation of input data (x) using a variety of
    basis functions, including polynomial, radial basis function (RBF),
    sigmoid, and Fourier transformations.
    Users can specify the basis function set_type and its parameters at
    initialization. The form of the output is design matrix (Φ).

    Design Matrix:
    Φ = [ [ φ_0(x_1), φ_1(x_1), ..., φ_M(x_1) ],
          [ φ_0(x_2), φ_1(x_2), ..., φ_M(x_2) ],
          ...
          [ φ_0(x_N), φ_1(x_N), ..., φ_M(x_N) ] ]

    where:
        - x_i denotes the i-th input vector.
        - φ_j(x_i) represents the j-th base function applied to the i-th input
            vector.
        - M is the total number of base functions (without φ_0(x_i)=1).
        - N is the total number of input vectors.

    Args:
        function_type (str): Type of the basis function to use
            ('polynomial', 'rbf', 'sigmoid', 'fourier').
        **kwargs: Additional parameters for the basis function. The expected
            parameters depend on the chosen basis function.

    Supported `kwargs` for each basis function include:
        - For 'rbf': 'n_centers' (int, default=10),
            'bandwidth' (float, default=1.0).

    Usage Example:
        >>> transformer = BasisFunctionTransform(
            'rbf', n_centers=15, bandwidth=0.5,
        )
        >>> transformer.preprocess(data)
        >>> transformed_data = transformer.transform(data)
    """

    def __init__(
        self,
        function_type: BasisFunctionType = BasisFunctionType.WITHOUT,
        with_bias: bool = True,
        **kwargs,
    ) -> None:
        self._basis_function = self._get_basis_function(function_type)
        self._with_bias = with_bias
        self._function_params = kwargs
        self._params: Dict[str, int] = {}

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """Make preprocessing for the chosen basis function if needed.

        Args:
            features: The input data to transform, 2D array (Nxd) with each row
                represents an input vector. N is number of elements in set,
                d is the size of feature vector.

        Returns:
            np.ndarray: The transformed data.
        """
        if self._basis_function is self._apply_rbf:
            return self._basis_function(features, preprocess=True)
        return self._basis_function(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Make feature transformation.

        Args:
            features: The input data to transform, 2D array (Nxd) with each row
                represents an input vector. N is number of elements in set,
                d is the size of feature vector.

        Returns:
            np.ndarray: The transformed data.
        """
        return self._basis_function(features)

    def _get_basis_function(
        self, function_type: BasisFunctionType,
    ) -> Callable:
        """Return the basis function object."""
        if function_type == BasisFunctionType.POLYNOMIAL:
            return self._apply_polynomial
        if function_type == BasisFunctionType.RBF:
            return self._apply_rbf
        if function_type == BasisFunctionType.SIGMOID:
            return self._apply_sigmoid
        if function_type == BasisFunctionType.FOURIER:
            return self._apply_fourier
        if function_type == BasisFunctionType.WITHOUT:
            return self._apply_default
        raise ValueError(f'Unknown basis function type: {function_type}')

    def _apply_polynomial(self, features: np.ndarray) -> np.ndarray:
        """Perform polynomial transformation on the input data.

        Transformation formula for input vector x with features
        [x1, x2, ..., xd] and max degree is: [1, x1, x2, ..., xd, x1^2, x2^2,
        ..., xd^2, ..., xd^degree]

        Keyword arguments (specified in self.function_params):
            'max_degree' (int): The maximum degree for the polynomial features.
                Determines how many polynomial terms are generated for each
                feature.
            'bias' (bool): If True, a column of ones (bias term) is added to
                the transformed data. Default is True.

        Args:
            features: The input data to transform, 2D array (Nxd) with each row
                represents an input vector. N is number of elements in set, d
                is the size of feature vector

        Returns:
            transformed_x (numpy.ndarray): transformed array. The size of an
                array: Nx(d*degree) if bias false or Nx(d*degree+1) if bias
                true.
        """
        # TODO:
        #  1) Get max_degree parameter from self.function_params
        #     >>> self.function_params['max_degree']
        #     Note that if max_degree is not specified you should use default.
        #  2) Create an empty list transformed_x
        #  3) Create a loop to generate polynomial features up to 'max_degree'
        #       for each feature in the input data.
        #     - The loop should start at 1 (since degree 0 is handled by the
        #           bias term if included) and go up to 'max_degree'.
        #     - Within the loop, raise each feature in the input to the current
        #           degree and append it to the transformed_x list.
        #       Note that you do not need to use a loop to raise each feature
        #           to the current degree, use numpy array feature.
        #  4) Check the bias parameter. 'bias' parameter should be specified in
        #       self.function_params:
        #     >>> self.function_params['bias']
        #     If 'bias' parameter is not specified the default value is True
        #       (a column of ones is added to the transformed_x array).
        #  5) Return transformed_x array.
        return features

    def _apply_rbf(
        self, features: np.ndarray, need_preprocessing: bool = False,
    ) -> np.ndarray:
        """Perform Radial Basis Function (RBF) transformation.

        The RBF is defined as:
            RBF(x, c) = exp(-||x - c||^2 / (σ^2))
        where 'x' is an input data point, 'c' is a center, and 'σ' (sigma) is
        the bandwidth.

        Keyword arguments (specified in self.function_params):
            'n_centers' (int): The number of centers to use for the RBF
                transformation. Default is 10. This is only used if
                'preprocess' is True.
            'bandwidth' (float): The bandwidth (σ) of the RBF kernel.
                Default is 1.0.
            'bias' (bool): If True, a column of ones (bias term) is added to
                the transformed data. Default is True.

        Args:
            x: The input data to transform, 2D array (Nxd) with each row
                represents an input vector. N is number of elements in set, d
                is the size of feature vector.
            preprocess: Flag to indicate if the operation is during training.
                   If True, the function selects new centers randomly.
                   Defaults to False.

        Returns:
            transformed_x (numpy.ndarray): The RBF-transformed array. The size
                of an array: Nxn_centers if bias False or Nx(n_centers+1) if
                bias True.

        Note: It is critical to run this function with preprocess=True at
            least once before using it to transform data, to ensure that
            centers are properly initialized.
        """
        # TODO
        #  1) Check the value of the 'preprocess' variable:
        #     -- If 'preprocess' is True:
        #       1) Randomly select a subset of the training data to be the
        #           centers.
        #          The number of centers is specified in the
        #           self.function_params:
        #              >>> self.function_params['centers']
        #          Please, ensure that the number of centers is not bigger
        #           than the number of data points in the subset.
        #          Note that if centers is not specified you should use
        #           default.
        #       2) Save selected centers in self.params. These centers would
        #           be used when preprocess is not False.
        #     -- If 'preprocess' is False:
        #       1) Get the previously selected centers stored in 'self.params':
        #              >>> self.params['centers']
        #          If self.params does not have key 'centers', raise
        #          ValueError("The RBF model is not trained. Please call the
        #           preprocess function before transform to initialize
        #           centers.").
        #  2) Calculate RBF:
        #     1) Compute the RBF transformation for each data point and each
        #           center with a specified bandwidth. The formula for RBF
        #           transformation is specified in the comments.
        #        The value of bandwidth is specified in self.function_params:
        #            >>> self.function_params['bandwidth']
        #        Note that if bandwidth is not specified you should use
        #           default.
        #     2) Save values in transformed_x variable.
        #  3) Check the bias parameter. 'bias' parameter should be specified
        #       in self.function_params:
        #     >>> self.function_params['bias']
        #     If 'bias' parameter is not specified the default value is True
        #       (a column of ones is added to the transformed_x array)
        #  4) Return transformed_x array.
        return features

    def _apply_sigmoid(self, features: np.ndarray) -> np.ndarray:
        """Perform Sigmoid transformation on the input data.

        The sigmoid function is defined as:
            sigmoid(x) = 1 / (1 + exp(-x))
        where 'x' is an input data point.

        Keyword arguments (specified in self.function_params):
            'bias' (bool): If True, a column of ones (bias term) is added to
            the transformed data. Default is True.

        Args:
            features: The input data to transform, 2D array (Nxd) with each row
                represents an input vector. N is number of elements in set,
                d is the size of feature vector.

        Returns:
            transformed_x (numpy.ndarray): The transformed data.
        """
        return features

    def _apply_fourier(self, features: np.ndarray) -> np.ndarray:
        """Perform Fourier transformation on the input data.

        The Fourier transformation is defined as:
            Fourier(x) = cos(x) + i * sin(x)
        where 'x' is an input data point.

        Keyword arguments (specified in self.function_params):
            'bias' (bool): If True, a column of ones (bias term) is added to
            the transformed data. Default is True.

        Args:
            features: The input data to transform, 2D array (Nxd) with each row
                represents an input vector. N is number of elements in set,
                d is the size of feature vector.

        Returns:
            transformed_x (numpy.ndarray): The transformed data.
        """
        return features

    def _apply_default(self, features: np.ndarray) -> np.ndarray:
        """The default transformation method applied to the input data.

        Perform a simple identity transformation on the input data:
            I(x) = x, where 'x' is an input data point.

        Keyword arguments (specified in self.function_params):
            'bias' (bool): If True, a column of ones (bias term) is added to
            the transformed data. Default is True.

        Args:
            features: The input data to transform, 2D array (Nxd) with each row
                represents an input vector. N is number of elements in set,
                d is the size of feature vector.

        Returns:
            transformed_x (numpy.ndarray): The transformed data.
        """
        # TODO
        #  1) Check the bias parameter. 'bias' parameter should be specified
        #     in self.function_params:
        #     >>> self.function_params['bias']
        #     If 'bias' parameter is not specified the default value is True
        #     (a column of ones is added to the input array x)
        #  2) Return x
        return features

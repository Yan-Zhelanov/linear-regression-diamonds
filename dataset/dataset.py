import os
import numpy as np
import pandas as pd

from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import Preprocessing


class DiamondsDataset:
    """A class for the Diamonds dataset. This class reads the data and preprocesses it."""

    def __init__(self, config):
        """Initializes the Diamonds dataset class instance."""
        self.config = config

        # Preprocessing class initialization
        self.preprocessing = Preprocessing(config.preprocess_type)

        # Reads the data
        self.data = {}
        for set_type in SetType:
            self.data[set_type.name] = self.dataframe_preprocessing(
                os.path.join(config.path_to_data, config.type[set_type.name]), set_type
            )

    def dataframe_preprocessing(self, path_to_dataframe: str, set_type: SetType) -> dict:
        """Preprocesses data.

        Args:
            path_to_dataframe: path to dataframe
            set_type: data set_type from SetType

        Returns:
            A dict with the following data: {'inputs: features (numpy.ndarray), 'targets': targets (numpy.ndarray)}
        """
        # TODO:
        #  1) Read a dataframe file using the read_dataframe_file(path_to_dataframe) function
        #  2) Drop duplicates from dataframe
        #  3) Convert categorical features to one-hot encoding vectors (columns ['color', 'clarity', 'cut'])
        #  4) Create features from all columns except 'price':
        #       - transform them to numpy.ndarray with dtype=np.float64
        #       - apply self.preprocessing according SetType:
        #              if set_type is SetType.train, call self.preprocessing.train(),
        #              otherwise call self.preprocessing()
        #  6) Create targets from columns 'price' (except when set_type is SetType.test):
        #       - transform to numpy.ndarray with dtype=np.float64
        #  7) Return arrays of features and targets as a dict
        raise NotImplementedError

    def __call__(self, set_type: str):
        """Returns preprocessed data."""
        return self.data[set_type]

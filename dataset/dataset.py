import os

import numpy as np
import pandas as pd

from config.data_config import Config
from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import Preprocessing


class DiamondsDataset(object):
    """A class for the Diamonds dataset.

    This class reads the data and preprocesses it.
    """

    def __init__(self, config: Config):
        """Initialize the Diamonds dataset class instance.

        Args:
            config: The data configuration.
        """
        self._config = config
        self._preprocessing = Preprocessing(config.PREPROCESS_TYPE)
        self._data = {}
        for set_type in SetType:
            self._data[set_type] = self.dataframe_preprocessing(
                os.path.join(config.PATH_TO_DATA, config.TYPE[set_type.name]),
                set_type,
            )

    def __call__(self, set_type: SetType):
        """Return preprocessed data."""
        return self._data[set_type]

    def dataframe_preprocessing(
        self, path_to_dataframe: str, set_type: SetType,
    ) -> dict:
        """Preprocesses data.

        Args:
            path_to_dataframe: path to dataframe
            set_type: data set_type from SetType

        Returns:
            dict: A dict with the following data:
                {'inputs: features (numpy.ndarray),
                'targets': targets (numpy.ndarray)}
        """
        df = read_dataframe_file(path_to_dataframe)
        df.drop_duplicates()
        df = pd.get_dummies(df, columns=['color', 'clarity', 'cut'])
        features = df.drop('price', axis=1).to_numpy(dtype=np.float64)
        if set_type is SetType.TRAIN:
            self._preprocessing.fit(features)
        else:
            features = self._preprocessing.preprocess(features)
        target = None
        if set_type is not SetType.TEST:
            target = df['price'].to_numpy(dtype=np.float64)
        return {'inputs': features, 'targets': target}

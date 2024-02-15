import pandas as pd
from typing import Union


def read_dataframe_file(path_to_file: str) -> Union[pd.DataFrame, None]:
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    else:
        raise ValueError("Unsupported file format. Only '.csv' and '.pickle' formats are supported.")

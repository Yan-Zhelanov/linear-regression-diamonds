import numpy as np
import pandas as pd

from config.data_config import Config
from dataset.dataset import DiamondsDataset
from model.linear_regression import LinearRegression
from utils.enums import SetType
from utils.metrics import get_rmse


def main():
    np.random.seed(0)

    dataset = DiamondsDataset(Config)
    train_data = dataset(SetType.TRAIN)
    valid_data = dataset(SetType.VALID)

    lin_reg_model = LinearRegression(
        number_bases=train_data['inputs'].shape[1] + 1,
    )
    lin_reg_model.fit(train_data['inputs'], train_data['targets'])

    train_predictions = lin_reg_model(train_data['inputs'])
    train_rmse = get_rmse(train_data['targets'], train_predictions)

    valid_predictions = lin_reg_model(valid_data['inputs'])
    valid_rmse = get_rmse(valid_data['targets'], valid_predictions)

    print(f'RMSE for train: {train_rmse}')
    print(f'RMSE for validation: {valid_rmse}')

    test_data = dataset('test')
    test_predictions = lin_reg_model(test_data['inputs'])

    test_results_df = pd.DataFrame(
        {'id': np.arange(0, len(test_predictions)), 'price': test_predictions},
    )
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    main()

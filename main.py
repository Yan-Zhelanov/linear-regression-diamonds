from dataset.dataset import DiamondsDataset
from model.linear_regression import LinearRegression
from config.data_config import data_config
from utils.metrics import RMSE
import numpy as np
import pandas as pd


def main():
    np.random.seed(0)

    dataset = DiamondsDataset(data_config)
    train_data = dataset('train')
    valid_data = dataset('valid')

    lin_reg_model = LinearRegression(m=train_data['inputs'].shape[1] + 1)
    lin_reg_model.train(train_data['inputs'], train_data['targets'])

    train_predictions = lin_reg_model(train_data['inputs'])
    train_rmse = RMSE(train_data['targets'], train_predictions)

    valid_predictions = lin_reg_model(valid_data['inputs'])
    valid_rmse = RMSE(valid_data['targets'], valid_predictions)

    print(f'RMSE for train: {train_rmse}')
    print(f'RMSE for validation: {valid_rmse}')

    test_data = dataset('test')
    test_predictions = lin_reg_model(test_data['inputs'])

    test_results_df = pd.DataFrame({'id': np.arange(0, len(test_predictions)), 'price': test_predictions})
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    main()

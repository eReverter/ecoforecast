"""
Enhanced script to train a regression model using LightGBM.
"""
import argparse
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.prepare_data import (
    load_data,
    add_is_weekend,
    get_surplus,
    get_forecast_target,
    convert_to_timeseries,
    get_lags,
    add_is_holiday,
    get_ohe_from_cat,
)
from src.definitions import MODELS_DIR, SEED, VAL_SIZE
from src.config import setup_logger

logger = setup_logger()

def prepare_data(df, lags):
    df = add_is_weekend(df)
    df = get_surplus(df)
    df = convert_to_timeseries(df, metadata_columns=['is_weekend'])
    df = get_lags(df, lags=lags)
    df = add_is_holiday(df)
    df = get_ohe_from_cat(df, cat='series_id')

    df = df.replace(0, np.nan)  # Replace 0s with NaNs
    df.dropna(inplace=True)  # Drop rows with NaNs created by lagging and shifting
    return df


def parser_add_arguments(parser):
    parser.add_argument('--use-grid', action='store_true', help='Use grid search for model tuning')
    return parser

def main():
    parser = argparse.ArgumentParser(description='Train LightGBM model')
    parser_add_arguments(parser)
    args = parser.parse_args()

    # Load data
    train, _, = load_data()

    # Data Preparation
    logger.info("Preparing data...")
    train = prepare_data(train, lags=[1,2,3])
    train = get_forecast_target(train, prediction_horizon=1)
    train.dropna(subset=['target'], inplace=True)

    x_train, x_val, y_train, y_val = train_test_split(
        train.drop(['timestamp', 'target', 'series_id'], axis=1),
        train['target'],
        test_size=VAL_SIZE,
        random_state=SEED
    )

    CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_config.json')
    if args.use_grid:
        # Grid Search
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            # Add more parameters here if desired
        }
        grid_search = GridSearchCV(
            estimator=lgb.LGBMRegressor(),
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=2,
            verbose=2
        )
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_

        # Save grid search best parameters to config
        with open(CONFIG_PATH, 'w') as f:
            json.dump(best_model.get_params(), f)
    else:
        # Load model configuration
        with open(CONFIG_PATH, 'r') as f:
            model_config = json.load(f)

        best_model = lgb.LGBMRegressor(**model_config)
        best_model.fit(x_train, y_train)

    # Validate model
    predictions = best_model.predict(x_val)
    val_mse = mean_squared_error(y_val, predictions)
    logger.info(f"Validation MSE: {val_mse}")

    # Save Model
    model_path = os.path.join(MODELS_DIR, 'forecasting', 'lightgbm', 'model.txt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_model.booster_.save_model(model_path)
    logger.info(f"Model trained and saved at {model_path}")

if __name__ == "__main__":
    main()


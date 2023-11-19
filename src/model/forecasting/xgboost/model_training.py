"""
Enhanced script to train a regression model using XGBoost.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import json
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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
from src.definitions import PROCESSED_DATA_DIR, MODELS_DIR, EXTERNAL_DATA_DIR, REGION_MAPPING
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
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser_add_arguments(parser)
    args = parser.parse_args()

    # Load data
    train, _, = load_data()

    # Load model configuration from this script directory
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_config.json')
    with open(CONFIG_PATH, 'r') as f:
        model_config = json.load(f)  # XGBoost configuration

    # Model path
    model_name = 'xgboost'
    model_path = os.path.join(f'{MODELS_DIR}/forecasting', model_name)

    # Data Preparation
    logger.info("Preparing data...")
    train = prepare_data(train, lags=[1,2,3])
    train = get_forecast_target(train, prediction_horizon=1)
    
    # Remove rows that have NaNs in the target column
    train.dropna(subset=['target'], inplace=True)

    # Prepare data for training in XGBoost
    x_train = train.drop(['timestamp', 'target', 'series_id'], axis=1)
    y_train = train['target']

    # if DEBUG:
    #     x_train = x_train.iloc[:1000]
    #     y_train = y_train.iloc[:1000]

    # Model Training
    logger.info("Training model...")
    model = xgb.XGBRegressor(**model_config)  # Switched to XGBRegressor
    model.fit(x_train, y_train)  # Add early stopping if needed

    # Feature Importance
    # logger.info("Displaying feature importance...")
    # xgb.plot_importance(model)
    # plt.show()

    # Saving Model
    os.makedirs(model_path, exist_ok=True)  # Simplified folder creation
    model.save_model(os.path.join(model_path, 'model.json'))

    logger.info(f"Model trained and saved at {model_path}")

if __name__ == "__main__":
    main()

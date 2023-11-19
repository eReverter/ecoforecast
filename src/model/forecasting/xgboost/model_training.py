"""
Enhanced script to train a model using XGBoost.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from src.utils import load_data
from src.data.prepare_data import (
    fill_time_series_gaps,
    impute_missing_values,
    add_metadata_features,
    get_lags,
    aggregate_to_freq,
)
from src.definitions import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR
from src.data.split_timeseries import (
    arrange_train_data,
    split_series_id,
    stratified_cv_series_id,
)
from src.data.metrics import weighted_normalized_mean_abs_error
from src.config import setup_logger

logger = setup_logger()

DEBUG = False

def parser_add_arguments(parser):
    parser.add_argument('--train_file', type=str, default='train.csv', help='File within the processed data folder with the training data')
    parser.add_argument('--test_file', type=str, default='test.csv', help='File within the processed data folder with the test data')
    parser.add_argument('--features_file', type=str, default='', help='File within the processed data folder with the features')
    parser.add_argument('--freq', type=str, default='1H', help='Frequency of the time series')
    parser.add_argument('--lags', type=lambda s: [int(item) for item in s.split(',')], default='1,3,7,12,24,48,168', help='List of lag values to use (comma-separated)')
    parser.add_argument('--cv', type=int, default=1, help='Number of cross validation folds')

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser_add_arguments(parser)
    args = parser.parse_args()

    # Set paths
    TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, args.train_file)
    TEST_PATH = os.path.join(PROCESSED_DATA_DIR, args.test_file)

    if len(args.features_file) == 0:
        FEATURES_PATH = None   
    else:
        FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, args.features_file)
        features = pd.read_csv(FEATURES_PATH)

    # Load data
    train, _, metadata, _ = load_data(TRAIN_PATH, TEST_PATH)

    # Load model configuration from this script directory
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)  # XGBoost configuration

    # Data Preprocessing
    logger.info("Preprocessing data...")
    train = fill_time_series_gaps(train, freq='1H')
    if DEBUG:
        print(train.head())
    train = impute_missing_values(train)
    if DEBUG:
        print(train.head())
    if FEATURES_PATH is not None:
        train = add_metadata_features(train, features)
        if DEBUG:
            print(train.head())
    train = aggregate_to_freq(train, freq=args.freq)
    if DEBUG:
        print(train.head())
    train = get_lags(train, lags=args.lags)
    if DEBUG:
        print(train.head())

    # Splitting Data
    logger.info("Splitting data...")
    unique_series_ids = train['series_id'].unique()
    if args.cv == 1:
        train_ids, val_ids = split_series_id(unique_series_ids, 0, 5)  # Assuming 20% for validation
    else:
        # Stratified CV not implemented in your script, assuming regular CV
        train_ids, val_ids = split_series_id(unique_series_ids, 0, args.cv, 0)

    # Prepare data for training in XGBoost
    x_train = train[train['series_id'].isin(train_ids)]
    x_train = x_train.drop(['series_id', 'timestamp'], axis=1)
    y_train = x_train.pop('consumption')

    x_val = train[train['series_id'].isin(val_ids)]
    x_val = x_val.drop(['series_id', 'timestamp'], axis=1)
    y_val = x_val.pop('consumption')

    if DEBUG:
        x_train = x_train.iloc[:1000]
        y_train = y_train.iloc[:1000]
        x_val = x_val.iloc[:1000]
        y_val = y_val.iloc[:1000]

    # Model Training
    logger.info("Training model...")
    model = xgb.XGBRegressor(**config)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=10)  # Add early stopping

    # Saving Model
    model_name = f'xgboost_freq:{args.freq}_lags:{args.lags}'
    model_path = os.path.join(CHECKPOINTS_DIR, model_name)
    os.makedirs(model_path, exist_ok=True)  # Simplified folder creation
    model.save_model(os.path.join(model_path, 'model.json'))

    logger.info(f"Model trained and saved at {model_path}")

if __name__ == "__main__":
    main()

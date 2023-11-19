"""
Enhanced script to train a classification model using XGBoost.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import json
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.prepare_data import (
    load_data,
    add_is_weekend,
    get_surplus,
    get_target,
)
from src.definitions import PROCESSED_DATA_DIR, MODELS_DIR
from src.config import setup_logger

logger = setup_logger()

def prepare_data(df):
    # Add weekend feature and compute surplus
    df = add_is_weekend(df)
    df = get_surplus(df)
    
    # Remove all cols that have 'load' and 'gen' in its names
    df = df.drop(df.filter(regex='load|gen').columns, axis=1)
    return df

def parser_add_arguments(parser):
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser_add_arguments(parser)
    args = parser.parse_args()

    # DEBUG = args.debug

    # Load data
    train, _, = load_data()

    # Load model configuration from this script directory
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_config.json')
    with open(CONFIG_PATH, 'r') as f:
        model_config = json.load(f)  # XGBoost configuration

    # Model path
    model_name = 'xgboost'
    model_path = os.path.join(MODELS_DIR, model_name)

    # Data Preparation
    logger.info("Preparing data...")
    train = prepare_data(train)
    train = get_target(train)

    # Encode target labels
    label_encoder = LabelEncoder()
    train['target'] = label_encoder.fit_transform(train['target'])

    # Store the label encoder for later use
    label_encoder_path = os.path.join(model_path, 'label_encoder.npy')
    np.save(label_encoder_path, label_encoder.classes_)

    # Prepare data for training in XGBoost
    x_train = train.drop(['timestamp', 'target'], axis=1)
    y_train = train['target']

    # if DEBUG:
    #     x_train = x_train.iloc[:1000]
    #     y_train = y_train.iloc[:1000]

    # Model Training
    logger.info("Training model...")
    model = xgb.XGBClassifier(**model_config)
    model.fit(x_train, y_train)  # Add early stopping

    # Feature Importance
    # logger.info("Displaying feature importance...")
    # xgb.plot_importance(model)
    # plt.show()

    # Saving Model
    os.makedirs(model_path, exist_ok=True)  # Simplified folder creation
    model.save_model(os.path.join(model_path, 'model.json'))

    logger.info(f"Model trained and saved at {model_path}")

    # Check the classes are being considered in the right order
    print(model.predict_proba(x_train.iloc[:1]))

if __name__ == "__main__":
    main()

"""
Script to train a classification model using XGBoost.
"""
# General imports
import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data related imports
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Local imports
from src.definitions import MODELS_DIR
from src.data.prepare_data import (
    load_data,
    add_is_weekend,
    get_surplus,
    get_curr_max,
    get_ohe_from_cat,
    get_cls_target,
)
from src.config import setup_logger

# Setup logger
logger = setup_logger()

### GENERAL FUNCTIONS ###
def prepare_data(df):
    """
    Prepare the data for training a classification  model.

    :param df: DataFrame with the data.
    :return: DataFrame with the data prepared.
    """
    # Add features
    df = add_is_weekend(df)
    df = get_surplus(df)
    df = get_curr_max(df)
    df = get_ohe_from_cat(df, cat='curr_max')
    
    # Remove all cols that have 'load' and 'gen' in its names. Only surplus will be used.
    df = df.drop(df.filter(regex='load|gen').columns, axis=1)
    return df

### MAIN ###

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
    model_path = os.path.join(MODELS_DIR, model_name)

    # Data Preparation
    logger.info("Preparing data...")
    train = prepare_data(train)
    train = get_cls_target(train)
    train = train.drop(['curr_max'], axis=1)

    # Encode target labels
    label_encoder = LabelEncoder()
    train['target'] = label_encoder.fit_transform(train['target'])

    # Store the label encoder for later use
    label_encoder_path = os.path.join(model_path, 'label_encoder.npy')
    np.save(label_encoder_path, label_encoder.classes_)

    # Prepare data for training in XGBoost
    x_train = train.drop(['timestamp', 'target'], axis=1)
    y_train = train['target']

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

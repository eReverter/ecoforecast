"""
Script to make predictions using an XGBoost trained model for classification.
"""
# General imports
import argparse
import os

# Data related imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Local imports
from src.definitions import PREDICTIONS_DIR
from src.data.prepare_data import load_data
from src.model.classification.xgboost.model_training import prepare_data

### GENERAL FUNCTIONS ###

def load_model(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

def main():
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--encoder', type=str, help='Path to encoder')
    parser.add_argument('--data', type=str, help='Path to prediction data')
    args = parser.parse_args()

    # Load Model
    model_path = args.model
    model = load_model(model_path)

    # Load and prepare prediction data
    _, validation = load_data()
    validation = prepare_data(validation)
    validation.drop(['curr_max'], axis=1, inplace=True)
    x_predict = validation.drop(['timestamp'], axis=1)

    # Make predictions
    predictions = model.predict(x_predict)

    # Reverse label encoding
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(args.encoder, allow_pickle=True)  # Load the saved classes
    decoded_predictions = label_encoder.inverse_transform(predictions)

    # Convert predictions to a df that has timestamp and target columns
    df = pd.DataFrame({'timestamp': validation['timestamp'], 'target': decoded_predictions})
    df[['timestamp', 'target']].to_json(os.path.join(PREDICTIONS_DIR, 'xgboost_cls_predictions.json'), orient='records')
    print("Predictions saved to predictions/")

if __name__ == "__main__":
    main()

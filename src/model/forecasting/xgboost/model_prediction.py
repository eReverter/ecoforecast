import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
import os
from src.data.prepare_data import load_data  # Adjust imports as needed
from src.model.forecasting.xgboost.model_training import prepare_data
from src.definitions import PREDICTIONS_DIR

def load_model(model_path):
    model = xgb.XGBRegressor()  # Make sure to use XGBRegressor
    model.load_model(model_path)
    return model

def main():
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--data', type=str, help='Path to prediction data')
    args = parser.parse_args()

    model_path = args.model
    model = load_model(model_path)

    # Load and prepare prediction data
    _, validation = load_data()
    validation = prepare_data(validation, lags=[1,2,3])
    x_predict = validation.drop(['timestamp', 'series_id'], axis=1)  # Drop non-feature columns

    # Make predictions
    predictions = model.predict(x_predict)

    # Convert predictions back to original format
    validation['predicted_surplus'] = predictions
    pivoted_df = validation.pivot(index='timestamp', columns='series_id', values='predicted_surplus')

    # Get the maximum country code for each row (timestamp)
    max_col = pivoted_df.idxmax(axis=1)
    pivoted_df['target'] = max_col

    predictions_path = os.path.join(PREDICTIONS_DIR, 'xgboost_reg_predictions.json')
    pivoted_df.reset_index()[['timestamp', 'target']].to_json(predictions_path, orient='records')

    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()

"""
Script to make predictions using an XGBoost trained model for regression.
"""
# General imports
import argparse
import os

# Data related imports
import lightgbm as lgb

# Local imports
from src.definitions import (
    PREDICTIONS_DIR,
    LIGHTGBM_LAGS,
)
from src.data.prepare_data import load_data
from src.model.forecasting.lightgbm.model_training import prepare_data

### GENERAL FUNCTIONS ###

def load_model(model_path):
    model = lgb.Booster(model_file=model_path)
    return model

### MAIN ###

def main():
    parser = argparse.ArgumentParser(description='Make predictions using LightGBM')
    parser.add_argument('--model', type=str, help='Path to model')
    args = parser.parse_args()

    model_path = args.model
    model = load_model(model_path)

    # Load and prepare prediction data
    _, validation = load_data()
    validation = prepare_data(validation, lags=LIGHTGBM_LAGS)
    x_predict = validation.drop(['timestamp', 'series_id'], axis=1)  # Drop non-feature columns

    # Make predictions
    predictions = model.predict(x_predict)

    # Convert predictions back to original format
    validation['predicted_surplus'] = predictions
    pivoted_df = validation.pivot(index='timestamp', columns='series_id', values='predicted_surplus')

    # Get the maximum country code for each row (timestamp)
    max_col = pivoted_df.idxmax(axis=1)
    pivoted_df['target'] = max_col

    predictions_path = os.path.join(PREDICTIONS_DIR, 'lightgbm_reg_predictions.json')
    pivoted_df.reset_index()[['timestamp', 'target']].to_json(predictions_path, orient='records')

    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()

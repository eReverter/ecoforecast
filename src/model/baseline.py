
import argparse
from src.definitions import (
    PROCESSED_DATA_DIR, 
    EXTERNAL_DATA_DIR,
    REGION,
    REGION_MAPPING,
    PREDICTIONS_DIR)
import pandas as pd
import os
from src.data.prepare_data import (
    get_surplus,
)

def naive_prediction(df):
    """
    Use current maximum surplus as the prediction for the next hour.
    """
    # Get the maximum surplus for each region
    df = get_surplus(df)
    # Get the target
    max_col = df.filter(regex='surplus').idxmax(axis=1)
    country_code = max_col.str.split('_', expand=True)[0]
    df['target'] = country_code

    # Store the target to a .json file called baseline.json
    df[['timestamp', 'target']].to_json(os.path.join(PREDICTIONS_DIR, 'baseline.json'), orient='records')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute baseline predictions')
    parser.add_argument('--data', type=str, help='Path to the data file')
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.data, parse_dates=['timestamp'])

    # Make the prediction
    naive_prediction(df)
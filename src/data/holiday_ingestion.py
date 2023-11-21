"""
Script to ingest holidays dates for each region and save them in a CSV file.
"""
# Genereal imports
import argparse
import holidays

# Data related imports
import pandas as pd

# Local imports
from src.definitions import (
    EXTERNAL_DATA_DIR, 
    REGION, 
    REGION_MAPPING
)

### MAIN ###

def add_parser_args(parser):
    parser.add_argument('--start_year', type=int, default=2022, help='Start year for holiday generation')
    parser.add_argument('--end_year', type=int, default=2023, help='End year for holiday generation')

def main():
    parser = argparse.ArgumentParser(description='Generate holidays for each region')
    add_parser_args(parser)
    args = parser.parse_args()

    for country, _ in REGION.items():
        # Generate holidays for 2022 and 2023
        country = REGION_MAPPING[country]
        country_holidays = holidays.CountryHoliday(country, years=[args.start_year, args.end_year])

        # Convert to DataFrame
        holiday_df = pd.DataFrame.from_dict(country_holidays, orient='index', columns=['Holiday'])
        holiday_df.reset_index(inplace=True)
        holiday_df.rename(columns={'index': 'Date'}, inplace=True)

        # Save to CSV
        holiday_df.to_csv(f'{EXTERNAL_DATA_DIR}/{country}_holidays.csv', index=False)
        print(f'Holidays for {country} saved in {country}_holidays.csv and {country}_holidays.json')
# General imports
import argparse
import os
from tqdm import tqdm

# Data related imports
import datetime
import pandas as pd

# Local imports
from src.utils import (
    perform_get_request, 
    xml_to_load_dataframe, 
    xml_to_gen_data
)

from src.config import setup_logger

# Setup logger
logger = setup_logger()

### GENERAL FUNCTIONS ###

def process_data_in_chunks(url, params, region, area_code, start, end, chunk_size, xml_to_dataframe, output_path, data_type):
    """
    Process data in chunks and save it to a CSV file.

    :param url: URL of the API.
    :param params: Parameters for the API.
    :param region: Region to fetch data from.
    :param area_code: Area code of the region.
    :param start: Start time of the data to fetch.
    :param end: End time of the data to fetch.
    :param chunk_size: Size of the chunks to fetch.
    :param xml_to_dataframe: Function to convert XML to DataFrame.
    :param output_path: Path to save the CSV files.
    :param data_type: Type of data to fetch (e.g., 'load', 'gen').
    """
    current_start = datetime.datetime.strptime(start, '%Y%m%d%H%M')
    end = datetime.datetime.strptime(end, '%Y%m%d%H%M')

    while current_start < end:
        # Define filename
        filename = f'{output_path}/{data_type}_{region}_{current_start.strftime("%Y%m%d")}.csv'

        # If file exists, skip
        if os.path.exists(filename):
            current_start += chunk_size
            continue
        
        # Log progress
        logger.info(f'Fetching {data_type} data from {region} for the period {current_start} - {min(current_start + chunk_size, end)}...')
        current_end = min(current_start + chunk_size, end)

        # Update params for the current chunk
        params['periodStart'] = current_start.strftime('%Y%m%d%H%M')
        params['periodEnd'] = current_end.strftime('%Y%m%d%H%M')
        params['outBiddingZone_Domain'] = area_code
        params['in_Domain'] = area_code

        # Fetch and process data
        response_content = perform_get_request(url, params)
        data = xml_to_dataframe(response_content)

        # Check and concatenate the DataFrame
        if data_type == 'load': # Single df
            if data is not None and not data.empty:
                data.to_csv(filename, index=False)

        elif data_type == 'gen': # Dict of df
            all_data = pd.DataFrame()
            for psr_type, df in data.items():
                all_data = pd.concat([all_data, df], ignore_index=True)

            if all_data is not None and not all_data.empty:
                all_data.to_csv(filename, index=False)

        # Move to the next chunk
        current_start = current_end

def get_load_data_from_entsoe(regions, start_time, end_time, output_path):
    # URL and general parameters for the Load data API
    url = 'https://web-api.tp.entsoe.eu/api'
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A65',
        'processType': 'A16'
    }

    for region, area_code in tqdm(regions.items(), desc="Fetching load data"):
        logger.info(f'Fetching load data from {region} for the period {start_time} - {end_time}...')
        process_data_in_chunks(url, params, region, area_code, start_time, end_time, datetime.timedelta(days=365), xml_to_load_dataframe, output_path, 'load')

def get_gen_data_from_entsoe(regions, start_time, end_time, output_path):
    # URL and general parameters for the Generation data API
    url = 'https://web-api.tp.entsoe.eu/api'
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A75',
        'processType': 'A16'
    }

    for region, area_code in tqdm(regions.items(), desc="Fetching gen data"):
            logger.info(f'MAIN: Fetching gen data from {region} for the period {start_time} - {end_time}...')
            process_data_in_chunks(url, params, region, area_code, start_time, end_time, datetime.timedelta(days=365), xml_to_gen_data, output_path, 'gen')

### MAIN ###

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data ingestion script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--start_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2022, 1, 1), 
        help='Start time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--end_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2022, 12, 31), 
        help='End time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./data/raw',
        help='Name of the output file'
    )
    return parser.parse_args()

def main(start_time, end_time, output_path):
    
    regions = {
        'HU': '10YHU-MAVIR----U',
        'IT': '10YIT-GRTN-----B',
        'PO': '10YPL-AREA-----S',
        'SP': '10YES-REE------0',
        'UK': '10Y1001A1001A92E',
        'DE': '10Y1001A1001A83F',
        'DK': '10Y1001A1001A65H',
        'SE': '10YSE-1--------K',
        'NE': '10YNL----------L',
    }

    # Transform start_time and end_time to the format required by the API: YYYYMMDDHHMM
    start_time = start_time.strftime('%Y%m%d%H%M')
    end_time = end_time.strftime('%Y%m%d%H%M')

    # Get Load data from ENTSO-E
    get_load_data_from_entsoe(regions, start_time, end_time, output_path)

    # Get Generation data from ENTSO-E
    get_gen_data_from_entsoe(regions, start_time, end_time, output_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.start_time, args.end_time, args.output_path)

    
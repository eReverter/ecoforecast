"""
Metrics for the end-to-end pipeline.
"""
import numpy as np
import json
from termcolor import colored
import pandas as pd

class StatisticsETL:
    def __init__(self, start_date=None, end_date=None):
        """
        Initializes the StatisticsETL object with start and end dates for the ETL process.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.original_count = {}
        self.green_energy_count = {}
        self.imputed_count = {}
        self.aggregated_count = {}
        self.outlier_count = {}
        self.na_count = {}
        self.estimated_frequency = {}

    def update_original_count(self, type_region, df):
        """
        Update the original row count for a given type and region.
        """
        self.original_count[type_region] = df.shape[0]

    def update_green_energy_count(self, type_region, df):
        """
        Update the count of green energy rows for a given type and region.
        """
        self.green_energy_count[type_region] = df.shape[0]

    def update_imputed_count(self, type_region, na_count):
        """
        Update the count of imputed rows for a given type and region.
        """
        self.imputed_count[type_region] = na_count

    def update_aggregated_count(self, type_region, df):
        """
        Update the count of aggregated rows for a given type and region.
        """
        self.aggregated_count[type_region] = df.shape[0]

    def update_na_count(self, type_region, na_count):
        """
        Update the count of NA values for a given type and region.
        """
        self.na_count[type_region] = na_count

    def update_outlier_count(self, type_region, outlier_count):
        """
        Update the count of identified outliers for a given type and region.
        """
        self.outlier_count[type_region] = outlier_count

    def update_estimated_frequency(self, type_region, freq):
        """
        Update the estimated frequency for a given type and region.
        """
        self.estimated_frequency[type_region] = freq

    def generate_statistics(self):
        """
        Generate a comprehensive summary of the collected statistics.
        """
        summary = {
            'Original Data Count': self.original_count,
            'Green Energy Data Count': self.green_energy_count,
            'Imputed Data Count': self.imputed_count,
            'Aggregated Data Count': self.aggregated_count,
            'NA Count': self.na_count,
            'Outlier Count': self.outlier_count,
            'Estimated Frequency': self.estimated_frequency
        }
        return summary
    
    def display_statistics(self):
        """
        Display the statistics with color coding for each dataset.
        """
        for dataset, stats in self.generate_statistics().items():
            print(colored(f'{dataset}:', 'blue'))
            for type_region, count in stats.items():
                print(colored(f'  {type_region}: {count}', 'green'))

    def display_statistics_as_table(self):
        """
        Display the collected statistics as a formatted table.
        """
        # Create a dictionary to hold reformatted statistics
        reformatted_stats = {}

        # Loop through each category of statistics and reformat
        for stat_category, stats in self.generate_statistics().items():
            for type_region, count in stats.items():
                if type_region not in reformatted_stats:
                    reformatted_stats[type_region] = {}
                reformatted_stats[type_region][stat_category] = count

        # Convert to DataFrame for better visualization
        stats_df = pd.DataFrame.from_dict(reformatted_stats, orient='index')
        
        # Print the DataFrame
        print(stats_df)

    def save_statistics_to_json(self, file_path):
        """
        Save the collected statistics to a JSON file.
        """
        with open(file_path, 'w') as file:
            json.dump(self.generate_statistics(), file, indent=4)

    def __repr__(self):
        """
        Represent the StatisticsETL object as a string.
        """
        return f"<StatisticsETL from {self.start_date} to {self.end_date}>"

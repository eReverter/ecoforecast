from src.definitions import EXTERNAL_DATA_DIR, REGION

import pandas as pd
import holidays

mapping = {
    "SP": "ES",
    "UK": "GB",
    "DE": "DE",
    "DK": "DK",
    "HU": "HU",
    "SE": "SE",
    "IT": "IT",
    "PO": "PL",
    "NE": "NL",
}

for country, _ in REGION.items():
    # Generate holidays for 2022 and 2023
    country = mapping[country]
    country_holidays = holidays.CountryHoliday(country, years=[2022, 2023])

    # Convert to DataFrame
    holiday_df = pd.DataFrame.from_dict(country_holidays, orient='index', columns=['Holiday'])
    holiday_df.reset_index(inplace=True)
    holiday_df.rename(columns={'index': 'Date'}, inplace=True)

    # Save to CSV
    holiday_df.to_csv(f'{EXTERNAL_DATA_DIR}/{country}_holidays.csv', index=False)

    print(f'Holidays for {country} saved in {country}_holidays.csv and {country}_holidays.json')
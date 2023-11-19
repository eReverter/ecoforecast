import os

ROOT_DIR = os.path.dirname(os.path.join('..', os.path.dirname(os.path.realpath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')

GREEN_ENERGY = [
    "B01",  # Biomass
    "B09",  # Geothermal
    "B10",  # Hydro Pumped Storage
    "B11",  # Hydro Run-of-river and poundage
    "B12",  # Hydro Water Reservoir
    "B13",  # Marine
    "B15",  # Other renewable
    "B16",  # Solar
    "B18",  # Wind Offshore
    "B19"   # Wind Onshore
]

TYPE = [
    "load",
    "gen"
]

REGION = {
    "SP": 0, # Spain
    "UK": 1, # United Kingdom
    "DE": 2, # Germany
    "DK": 3, # Denmark
    "HU": 5, # Hungary
    "SE": 4, # Sweden
    "IT": 6, # Italy
    "PO": 7, # Poland
    "NE": 8 # Netherlands
}

REGION_MAPPING = {
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

SEED = 3
VAL_SIZE = 0.1

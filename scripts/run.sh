#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH


# python src/data/data_ingestion.py

python src/data/data_processing.py \
    --process_raw_data \
    --process_interim_data \
    --account_for_missing_intervals \
    --mode train
    
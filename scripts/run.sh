#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH


# python src/data/data_ingestion.py

# python src/data/data_processing.py \
#     --process_interim_data \
    # --process_raw_data \
    # --interpolate_zeros \
    # --process_interim_data \
#     --mode validation

python src/visualization/visualize.py \
    --mode train 

# python src/data/prepare_data.py \
#     --validation data/processed/validation.csv

# python src/model/baseline.py \
#     --data data/processed/validation.csv \

# python src/model/forecasting/lstm/model_training.py \
#     --use-grid \
#     --cnn \

# python src/model/forecasting/lstm/model_prediction.py \
#     --model models/forecasting/lstm/model.pth \
#     --cnn \
#     --data data/processed/validation.csv

# python src/metrics.py \
#     --predictions predictions/lstm_predictions.json
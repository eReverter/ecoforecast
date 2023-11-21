#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH


# python src/data/data_ingestion.py \
#     --start_time "2022-01-01" \
#     --end_time "2023-01-31" \
#     --output_path data/raw/train

# python src/data/data_processing.py \
#     --process_interim_data \
#     --mode train

# python src/visualization/visualize.py \
#     --mode train

# python src/visualization/visualize.py \
#     --mode validation

# python src/data/prepare_data.py \
#     --validation data/processed/validation.csv

# python src/model/baseline.py \
#     --data data/processed/validation.csv \

# python src/model/forecasting/lstm/model_training.py

# python src/model/forecasting/xgboost/model_training.py \
    # --use-grid \

# python src/model/forecasting/lstm/model_prediction.py \
#     --model models/forecasting/lstm/model.pth \
    
# python src/metrics.py \
    # --predictions predictions/lstm_predictions.json
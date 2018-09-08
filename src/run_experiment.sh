#!/bin/bash
python -O preprocessing_tweet_remote.py
python -O preprocessing_interpolated_gps.py
python -O preprocessing_interpolated_gps_evaluation.py
python -O prediction_gps_grid_mode_topic.py
python -O evaluate_prediction.py
python -O evaluate_prediction_cityemd.py
python -O utility_dropbox.py
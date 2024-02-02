#!/bin/bash
python -O preprocessing_1_embedding.py
python -O preprocessing_2_train_topic_modeling_local.py
python -O preprocessing_3_create_topic_features.py
python -O preprocessing_4_create_experiment_profile_table.py
python -O preprocessing_5_interploated_gps_db.py
python -O prediction_2_gps_grid_baseline_rnn.py
python -O prediction_3_gps_grid.py
python -O prediction_4_gps_grid_mode.py
python -O prediction_5_gps_grid_topic.py
python -O prediction_6_gps_grid_mode_topic.py
python -O evaluation_1_prediction.py
python -O evaluation_2_prediction_cityemd.py
python -O evaluation_3_prediction_small_target.py


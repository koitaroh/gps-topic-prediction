# Human mobility prediction with GPS trajectory and Twitter data

## Requirements:
- Python 3.7 or later
- MySQL
- Postgres
- config.cfg
- JUMAN++ or MeCab
    - For Japanese tweets


You'll need a configuration file "config.cfg" with MySQL/Postgres connection information (replace * with your keys).

```
[X/Twitter database]
host = ****
user = ****
passwd = ****
db_name = ****

[GPS trajectory database]
host = ****
user = ****
passwd = ****
db_name = ****
```

## Workflow

### (optional: tmux)
1. `tmux new -s prediction`
2. `tmux attach -t prediction`
3. control + b, d to detach

### Using Conda
(assuming Nvidia libraries are properly configured)
1. conda env create --name prediction --file environment.yml
2. conda activate prediction

### Run code

1. Review `settings.py`
    - Check settings and parameters of models.
    - What is your target area (in xy coordinates)?
    - What is your target training time span? (e.g. 2012-07-25 00:00:00 ~ 2012-07-31 23:59:59)
    - What is your target test time span? (e.g. 2012-07-25 08:00:00 ~ 2012-07-25 08:59:59) 
2. Preprocessing
    1. Embedding
        1. Run `preprocessing_1_embedding.py`
    2. X/Twitter topic modeling
        1. Run `preprocessing_2_train_topic_modeling_local.py`
        2. Run `preprocessing_3_create_topic_features.py`
    3. GPS (mobile phones)
        1. Run `preprocessing_4_create_experiment_profile_table.py`
            - Load GPS trajectory from DB and create profile table.
        1. Run `preprocessing_5_interpolated_gps_db.py`
            - Load GPS from DB and process data for machine learning model.
3. Run experiment (training and evaluation)
    1. Training
        1. Run `prediction_1_gps_grid_baseline_ngram.py` for runnning ngram baseline model.
        2. RUN `prediction_2_gps_grid_baseline_rnn.py` for running RNN baseline model.
        3. Run `prediction_3_gps_grid.py` for GPS only model.
        4. Run `prediction_4_gps_grid_mode.py` for GPS and mode model.
        5. Run `prediction_5_gps_grid_topic.py` for GPS and topic model.
        6. Run `prediction_6_gps_grid_mode_topic.py` for GPS, mode, and topic model.
        7. Run `prediction_7_gps_grid_mode_topic_attnention.py` for GPS, mode, and topic model with attention layer.
    2. Evaluation
        1. Run `evaluation_1_prediction.py`
            - Evaluate prediction with Loss
        2. Run `evaluation_2_prediction_cityemd.py`
            - Evaluate prediction with CityEMD
        3. Run `evaluation_3_prediction_small_target.py`
            - Evaluate prediction in small target area


## data

| directory     | Description                    |
| ------------- | ------------------------------ |
|DATA_DIR_RAW = "../data/raw/" | Raw data. Ignored. | 
|DATA_DIR_PROCESSED = "../data/processed/" | Processed data. Ignored. |
|DATA_DIR_INTERIM = "../data/interim/" | Interim data. Ignored. |
|OUTPUT_DIR = "../data/output/" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + "/" | Output data. Ignored, but uploaded to Dropbox. |
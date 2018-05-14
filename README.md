# Human mobility prediction with GPS and Twitter
=========

Human mobility prediction with GPS and Twitter
# src

| file name     | Description                    |
| ------------- | ------------------------------ |
| settings.py | setting files |
| preprocessing_gps.py | Preprocessing of GPS data|


# data

| directory     | Description                    |
| ------------- | ------------------------------ |
|DATA_DIR_RAW = "../data/raw/" | Raw data. Ignored. | 
|DATA_DIR_PROCESSED = "../data/processed/" | Processed data. Version tracked. |
|DATA_DIR_INTERIM = "../data/interim/" | Interim data. Ignored. |
|OUTPUT_DIR = "../data/output/" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + "/" | Output data. Ignored, but uploaded to Dropbox. |

# Requirements:
- Python 3.7 or later
- MySQL
- config.cfg
- JUMAN++ or MeCab
    - For Japanese tweets


# Workflow

## Set up environment
Refered Keras docker build.


## Run code

1. Review `settings.py`
    - Check settings and parameters of models.
    - What is your target area (in xy coordinates)?
    - What is your target training time span? (e.g. 2012-07-25 00:00:00 ~ 2012-07-31 23:59:59)
    - What is your target test time span? (e.g. 2012-07-25 08:00:00 ~ 2012-07-25 08:59:59) 
2. Preprocessing
    1. Embedding
        1. Run `preprocessing_embedding.py`
    2. Twitter
        1. Run `preprocessing_tweet.py`
        2. Run `preprocessing_tweet_evaluation.py`
    3. GPS (mobile phones)
        1. Run `preprocessing_gps.py`
            - Load raw GPS from directory and filter data based on timespan, AOI, number of users.
        2. Run `preprocessing_interpolated_gps.py`
            - This will create training and test data.
        3. Run `preprocessing_interpolated_gps_evaluation.py`
            - This will create test data with experiment setting.
    4. GPS (app)
3. Run experiment (training and evaluation)
    1. Training
        1. Run `prediction_gps_grid_mode_topic.py` for main model.
        2. Run `prediction_gps_grid_baseline.py` for runnning baseline models.
    2. Evaluation
        1. Run `evaluate_prediction.py`
            - Evaluate prediction with Loss
        2. Run `evaluate_prediction_cityemd.py`
            - Evaluate prediction with CityEMD
    3. Upload results to Dropbox
        1. Run `utility.dropbox.py`
            - This will upload output trajectory and figures to Dropbox.
    
    
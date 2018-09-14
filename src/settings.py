import datetime
import os
import pymysql
import configparser
from slackclient import SlackClient
import sqlalchemy
from pathlib import Path

# Logging ver. 2017-12-14
import logging
from logging import handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('log.log', maxBytes=1000000, backupCount=3)  # file handler
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # console handler
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - [%(levelname)s][%(funcName)s] - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Initializing %s', __name__)

# # Database configuration
conf = configparser.ConfigParser()
src_dir = Path(__file__).parent.resolve()
# print(f'Checking the new pathlib: {src_dir}')
conf_file = src_dir / "config.cfg"
conf.read(conf_file)
SLACK_TOKEN = conf.get('Slack', 'token')
sc = SlackClient(SLACK_TOKEN)

######
# EXPERIMENT_ENVIRONMENT = "local"
EXPERIMENT_ENVIRONMENT = "remote"

SCENARIO = 'usual'
# SCENARIO = 'fireworks'
# SCENARIO = 'typhoon'
######


## Local
#####################################################################################
if EXPERIMENT_ENVIRONMENT == "local":
    EXPERIMENT_PARAMETERS = {
        'EXPERIMENT_NAME': "Experiment_local_201808281106",
        # 'TIMESTART' : "2012-07-19 00:00:00",
        # 'TIMEEND' : "2012-07-25 23:59:59",
        'AOI' : [138.72, 34.9, 140.87, 36.28],
        'UNIT_TEMPORAL' : 10,  # in minutes
        'UNIT_SPATIAL_METER' : 1000,
        # 'INPUT_DATASET' : "ZDC",
        'INPUT_DATASET' : "Interpolated",
        # 'INPUT_DATASET' : "iBank",
        'PREDICTION_INPUT_LENGTH': 144, # 144 for 24 hour in 10 mins
        'PREDICTION_OUTPUT_LENGTH': 6 , # 6 for 1 hour in 10 mins
        'SAMPLE_USER_SIZE': 500, # 107618 default
        'SAMPLE_SIZE' : 500, #1000000 for default, 10000000 for more
        # 'TRAINING_OBSERVATION_START': "2012-07-25 07:00:00",
        # 'TRAINING_OBSERVATION_END': "2012-07-26 06:59:59",
        # 'TRAINING_PREDICTION_START': "2012-07-26 07:00:00",
        # 'TRAINING_PREDICTION_END': "2012-07-26 07:59:59",

        'EVALUATION_SAMPLE_SIZE': 500,  # Sample size for visualization
        # 'EVALUATION_OBSERVATION_START': "2012-07-25 07:00:00",
        # 'EVALUATION_OBSERVATION_END': "2012-07-26 06:59:59",
        # 'EVALUATION_PREDICTION_START': "2012-07-26 07:00:00",
        # 'EVALUATION_PREDICTION_END': "2012-07-26 07:59:59",

        'VISUALIZATION_SAMPLE_SIZE': 500,  # Sample size for visualization
        'VISUALIZATION_CSV_SAMPLE_SIZE': 500,  # Sample size for visualization
    }

    # EXPERIMENT_DIR = Path("/Users/koitaroh/Documents/Data/Experiments/") / EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"]

    # DATA_EXTERNAL_DIR = src_dir.parent / "data/external"
    # DATA_INTERIM_DIR = src_dir.parent / "data/interim"
    # DATA_PROCESSED_DIR = src_dir.parent / "data/processed"
    # DATA_RAW_DIR = src_dir.parent / "data/raw"
    # RESULTS_DIR = src_dir.parent / "results/"

    DATA_DIR_RAW = src_dir.parent / "data/raw/"
    DATA_DIR_PROCESSED = src_dir.parent / "data/processed/"
    DATA_DIR_INTERIM = src_dir.parent / "data/interim/"
    OUTPUT_DIR = src_dir.parent / "data/output/" / EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"]

    if EXPERIMENT_PARAMETERS["INPUT_DATASET"] == "ZDC":
        GPS_RAW_DIR = DATA_DIR_RAW / "2012/"
    elif EXPERIMENT_PARAMETERS["INPUT_DATASET"] == "Interpolated":
        GPS_RAW_DIR = DATA_DIR_RAW / "gps_trip_result_kanto_201207/"

    GPS_FILTERED = DATA_DIR_PROCESSED / ("filtered_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    GPS_INTERPOLATED_FILTERED = DATA_DIR_PROCESSED / (
                "filtered_interpolated_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    GPS_INTERPOLATED_FILTERED_EVALUATION = DATA_DIR_PROCESSED / (
                "filtered_interpolated_evaluation_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    # TRANSFER_CSV_FILTERED = EXPERIMENT_DIR + "transfer_filtered.csv"


    # if EXPERIMENT_PARAMETERS["INPUT_DATASET"] == "ZDC":
    #     DATA_DIR = Path("/Users/koitaroh/Documents/Data/GPS/2012/")
    # elif EXPERIMENT_PARAMETERS["INPUT_DATASET"] == "Interpolated":
    #     DATA_DIR = Path("/Users/koitaroh/Documents/Data/GPS/gps_trip_result_kanto_201207/")
    # GPS_DIR_PROCESSED = Path("/Users/koitaroh/Documents/Data/GPS/zdc_filtered/")
    # GPS_FILTERED = GPS_DIR_PROCESSED / ("filtered_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    # GPS_INTERPOLATED_FILTERED = GPS_DIR_PROCESSED / ("filtered_interpolated_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    # GPS_INTERPOLATED_FILTERED_EVALUATION = GPS_DIR_PROCESSED / ("filtered_interpolated_evaluation_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    # # TRANSFER_CSV_FILTERED = EXPERIMENT_DIR + "transfer_filtered.csv"
    TEST_DB_TABLE_NAME = "gps_log_2"

    # STOPLIST_FILE = "../data/processed/stoplist_jp.txt"
    # MODE_LIST = ['BIKE','CAR','STAY','TRAIN','UNKNOWN','WALK']
    # MESHCODE_FILE = "../data/processed/Mesh3_Kanto_20171218_1221.csv"
    # LB_MODE_CLASSES_FILE = "../data/processed/lb_mode_classes.npy"
    # LE_GRID_CLASSES_FILE = "../data/processed/le_grid_classes.npy"

###################################################################

## Remote
#################################################################
if EXPERIMENT_ENVIRONMENT == "remote":
    EXPERIMENT_PARAMETERS = {
        'EXPERIMENT_NAME': "Experiment_fireworks_20180913_0820",
        # 'EXPERIMENT_NAME': "Experiment_20120725_20120725_5mins_25000users_2h_2layers",
        # 'TIMESTART' : "2012-07-19 00:00:00",
        # 'TIMEEND' : "2012-07-25 23:59:59",
        # aoi = [122.933198,24.045416,153.986939,45.522785] # Japan
        'AOI' : [138.72, 34.9, 140.87, 36.28],
        'UNIT_TEMPORAL' : 10,  # in minutes
        'UNIT_SPATIAL_METER' : 1000,
        # 'INPUT_DATASET' : "ZDC",
        'INPUT_DATASET' : "Interpolated",
        # 'INPUT_DATASET' : "iBank",
        'PREDICTION_INPUT_LENGTH': 144, # 144 for 12 hour in 10 mins
        'PREDICTION_OUTPUT_LENGTH': 6 , # 6 for 1 hour in 10 mins
        'SAMPLE_USER_SIZE': 30000, # 10000 for 456, 100000 for dpdl4
        'SAMPLE_SIZE' : 30000, #10000 for 456, 100000 for dpdl4

        # 'TRAINING_OBSERVATION_START': "2012-07-25 07:00:00",
        # 'TRAINING_OBSERVATION_END': "2012-07-26 06:59:59",
        # 'TRAINING_PREDICTION_START': "2012-07-26 07:00:00",
        # 'TRAINING_PREDICTION_END': "2012-07-26 07:59:59",


        'EVALUATION_SAMPLE_SIZE': 30000,  # Sample size for visualization
        # 'EVALUATION_OBSERVATION_START': "2012-07-25 07:00:00",
        # 'EVALUATION_OBSERVATION_END': "2012-07-26 06:59:59",
        # 'EVALUATION_PREDICTION_START': "2012-07-26 07:00:00",
        # 'EVALUATION_PREDICTION_END': "2012-07-26 07:59:59",

        'VISUALIZATION_SAMPLE_SIZE': 200,  # Sample size for visualization
        'VISUALIZATION_CSV_SAMPLE_SIZE': 30000,  # Sample size for visualization

        # Transfer
        # 'TIMESTART': "2016-06-01 07:00:00",
        # 'TIMEEND': "2016-06-01 18:59:59",
        # 'TIMESTART_TEXT': "2016-06-01",
        # 'TIMEEND_TEXT': "2016-06-01",
        # 'SAMPLE_SIZE' : 10000, #1000000 for default
    }

    DATA_DIR_RAW = Path('/data/miyazawa/')
    DATA_DIR_PROCESSED = src_dir.parent / "data/processed/"
    DATA_DIR_INTERIM = src_dir.parent / "data/interim/"
    OUTPUT_DIR = src_dir.parent / "data/output/" / EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"]

    if EXPERIMENT_PARAMETERS["INPUT_DATASET"] == "ZDC":
        GPS_RAW_DIR = DATA_DIR_RAW / "2012/"
    elif EXPERIMENT_PARAMETERS["INPUT_DATASET"] == "Interpolated":
        GPS_RAW_DIR = DATA_DIR_RAW / "gps_trip_result_kanto_201207/"

    GPS_FILTERED = DATA_DIR_PROCESSED / ("filtered_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    GPS_INTERPOLATED_FILTERED = DATA_DIR_PROCESSED / (
                "filtered_interpolated_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    GPS_INTERPOLATED_FILTERED_EVALUATION = DATA_DIR_PROCESSED / (
                "filtered_interpolated_evaluation_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")
    # TRANSFER_CSV_FILTERED = EXPERIMENT_DIR + "transfer_filtered.csv"

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    TEST_DB_TABLE_NAME = "gps_log_2"
#################################################################


# Scenarios
if SCENARIO == 'usual':
    EXPERIMENT_PARAMETERS['TIMESTART'] = '2012-07-19 00:00:00'
    EXPERIMENT_PARAMETERS['TIMEEND'] = '2012-07-25 23:59:59'
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START'] = "2012-07-25 08:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END'] = "2012-07-26 07:59:59"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START'] = "2012-07-26 08:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END'] = "2012-07-26 08:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_START'] = "2012-07-25 08:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_END'] = "2012-07-26 07:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_START'] = "2012-07-26 08:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_END'] = "2012-07-26 08:59:59"
elif SCENARIO == 'fireworks':
    EXPERIMENT_PARAMETERS['TIMESTART'] = '2012-07-19 00:00:00'
    EXPERIMENT_PARAMETERS['TIMEEND'] = '2012-07-25 23:59:59'
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START'] = "2012-07-27 19:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END'] = "2012-07-28 18:59:59"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START'] = "2012-07-28 19:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END'] = "2012-07-28 19:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_START'] = "2012-07-27 19:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_END'] = "2012-07-28 18:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_START'] = "2012-07-28 19:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_END'] = "2012-07-28 19:59:59"
elif SCENARIO == 'typhoon':
    EXPERIMENT_PARAMETERS['TIMESTART'] = '2012-07-19 00:00:00'
    EXPERIMENT_PARAMETERS['TIMEEND'] = '2012-07-25 23:59:59'


STOPLIST_FILE = "../data/processed/stoplist_jp.txt"
MODE_LIST = ['BIKE','CAR','STAY','TRAIN','UNKNOWN','WALK']
MESHCODE_FILE = "../data/processed/Mesh3_Kanto_20171218_1221.csv"
MESHCODE_FILE_2ND = "../data/processed/Mesh2_Kanto_20180111_1602.csv"
CITYEMD_DISTANCE_MATRIX_FILE = "../data/processed/cityemd_distance_matrix.npy"
LB_MODE_CLASSES_FILE = "../data/processed/lb_mode_classes.npy"
LE_GRID_CLASSES_FILE = "../data/processed/le_grid_classes.npy"


FIGURE_DIR = OUTPUT_DIR / "figures/"
TRAINING_DIR = OUTPUT_DIR / "training/"
EVALUATION_DIR = OUTPUT_DIR / "evaluation/"

logger.info("EXPERIMENT_NAME: %s", EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"])
logger.info("EXPERIMENT PARAMETERS: %s" % EXPERIMENT_PARAMETERS)
if not os.path.exists(DATA_DIR_PROCESSED): os.makedirs(DATA_DIR_PROCESSED)
if not os.path.exists(DATA_DIR_INTERIM): os.makedirs(DATA_DIR_INTERIM)
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(FIGURE_DIR): os.makedirs(FIGURE_DIR)
if not os.path.exists(TRAINING_DIR): os.makedirs(TRAINING_DIR)
if not os.path.exists(EVALUATION_DIR): os.makedirs(EVALUATION_DIR)

### File location for city-scale prediction
TEMPORAL_DATAFRAME = DATA_DIR_INTERIM / "dataframe.h5"

X_COORDINATE_FILE = DATA_DIR_INTERIM / "x_coordinate.npy"
Y_COORDINATE_FILE = DATA_DIR_INTERIM / "y_coordinate.npy"
X_MODE_FILE = DATA_DIR_INTERIM / "x_mode.npy"
X_GRID_FILE = DATA_DIR_INTERIM / "x_grid.npy"
Y_GRID_FILE = DATA_DIR_INTERIM / "y_grid.npy"
Y_MODE_FILE = DATA_DIR_INTERIM / "y_mode.npy"
X_TOPIC_FILE = DATA_DIR_INTERIM / "x_topic.npy"
Y_TOPIC_FILE = DATA_DIR_INTERIM / "y_topic.npy"

X_COORDINATE_EVALUATION_FILE = DATA_DIR_INTERIM / "evaluation_x_coordinate.npy"
Y_COORDINATE_EVALUATION_FILE = DATA_DIR_INTERIM / "evaluation_y_coordinate.npy"
X_MODE_EVALUATION_FILE = DATA_DIR_INTERIM / "evaluation_x_mode.npy"
X_GRID_EVALUATION_FILE = DATA_DIR_INTERIM / "evaluation_x_grid.npy"
Y_GRID_EVALUATION_FILE = DATA_DIR_INTERIM / "evaluation_y_grid.npy"
Y_MODE_EVALUATION_FILE = DATA_DIR_INTERIM / "evaluation_y_mode.npy"
X_TOPIC_EVALUATION_FILE = DATA_DIR_INTERIM / "evaluation_x_topic.npy"
Y_TOPIC_EVALUATION_FILE = DATA_DIR_INTERIM / "evaluation_y_topic.npy"

X_FILE = DATA_DIR_INTERIM / "x.npy"
Y_FILE = DATA_DIR_INTERIM / "y.npy"
Y_FILE_PREDICTED_LSTM = DATA_DIR_INTERIM / "y_predicted_lstm.npy"
Y_FILE_PREDICTED_VELOCITY = DATA_DIR_INTERIM / "y_predicted_laststep.npy"

MODEL_FILE_TRANSFER_LSTM = DATA_DIR_INTERIM / "model_transfer_lstm.h5"
MODEL_FILE_LSTM = DATA_DIR_INTERIM / "model_lstm.h5"
MODEL_WEIGHT_FILE_LSTM = DATA_DIR_INTERIM / "model_weights_lstm.h5"
MODEL_FILE_LSTM_GRID = DATA_DIR_INTERIM / "model_lstm_grid.h5"
MODEL_WEIGHT_FILE_LSTM_GRID = DATA_DIR_INTERIM / "model_weights_lstm_grid.h5"

GEOJSON_FILE_OBSERVATION = TRAINING_DIR / "trajectory_coordinate_observation.geojson"
GEOJSON_FILE_TRUE = TRAINING_DIR / "trajectory_coordinate_true.geojson"
GEOJSON_FILE_PREDICTED_LSTM = TRAINING_DIR / "trajectory_coordinate_predicted_lstm.geojson"
GEOJSON_FILE_OBSERVATION_GRID = TRAINING_DIR / "trajectory_grid_observation.geojson"
GEOJSON_FILE_TRUE_GRID = TRAINING_DIR / "trajectory_grid_true.geojson"
GEOJSON_FILE_PREDICTED_LSTM_GRID = TRAINING_DIR / "trajectory_grid_predicted_lstm.geojson"

GEOJSON_FILE_EVALUATION_OBSERVATION = EVALUATION_DIR / "trajectory_coordinate_observation.geojson"
GEOJSON_FILE_EVALUATION_TRUE = EVALUATION_DIR / "trajectory_coordinate_true.geojson"
GEOJSON_FILE_EVALUATION_PREDICTED_LSTM = EVALUATION_DIR / "trajectory_coordinate_predicted_lstm.geojson"
GEOJSON_FILE_EVALUATION_OBSERVATION_GRID = EVALUATION_DIR / "trajectory_grid_observation.geojson"
GEOJSON_FILE_EVALUATION_TRUE_GRID = EVALUATION_DIR / "trajectory_grid_true.geojson"
GEOJSON_FILE_EVALUATION_PREDICTED_LSTM_GRID = EVALUATION_DIR / "trajectory_grid_predicted_lstm.geojson"

# GEOJSON_FILE_EVALUATION_RAW = EVALUATION_DIR + "trajectory_coordinate_raw.geojson"

CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID = EVALUATION_DIR / "csv_trajectory_grid_observation.csv"
CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID = EVALUATION_DIR / "csv_trajectory_grid_true.csv"
CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID = EVALUATION_DIR / "csv_trajectory_grid_predicted.csv"

CSV_TRAJECTORY_FILE_EVALUATION_RAW = EVALUATION_DIR / "csv_trajectory_raw.csv"

LSI_MODEL_FILE = str(DATA_DIR_INTERIM / "Topic_LSI.model")
LDA_MODEL_FILE = str(DATA_DIR_INTERIM / "Topic_LDA.model")
DOC2VEC_MODEL_FILE = str(DATA_DIR_INTERIM / "Topic_Doc2Vec.model")

LSI_TOPIC_FILE = str(DATA_DIR_PROCESSED / "Topic_Feature_LSI.npy")
LDA_TOPIC_FILE = str(DATA_DIR_PROCESSED / "Topic_Feature_LDA.npy")
DOC2VEC_TOPIC_FILE = str(DATA_DIR_PROCESSED / "Topic_Feature_Doc2Vec.npy")

LSI_TOPIC_EVALUATION_FILE = str(DATA_DIR_PROCESSED / "Topic_Feature_evaluation_LSI.npy")
LDA_TOPIC_EVALUATION_FILE = str(DATA_DIR_PROCESSED / "Topic_Feature_evaluation_LDA.npy")
DOC2VEC_TOPIC_EVALUATION_FILE = str(DATA_DIR_PROCESSED / "Topic_Feature_evaluation_Doc2Vec.npy")

DICT_FILE = str(DATA_DIR_INTERIM / "corpus.dict")
MM_CORPUS_FILE = str(DATA_DIR_INTERIM / "corpus.mm")
TFIDF_FILE = str(DATA_DIR_INTERIM / "tfidf.tfidf")


# TEMPORAL_DATAFRAME = DATA_DIR / "interim/temp.h5"
#
# ZDC_RAW_DIR = DATA_DIR / "raw/zdc_raw/"
# ZDC_FILTERED_DIR = DATA_DIR / "processed/zdc_filtered/"
# ZDC_FILTERED_FILL5MINS_DIR = DATA_DIR / "processed/zdc_5mins_fill/"
# ZDC_FILTERED_DIR_1 = DATA_DIR / "processed/zdc_stay_1/"
# ZDC_FILTERED_DIR_2 = DATA_DIR / "processed/zdc_stay_2/"
# ZDC_FILTERED_DIR_3 = DATA_DIR / "processed/zdc_stay_3/"
# ZDC_FILTERED_DIR_4 = DATA_DIR / "processed/zdc_stay_4/"
# ZDC_STAYPOINT_STAT_FILE = DATA_DIR / "raw/staypoint_stat.csv"
# ZDC_FILTERED_FILE = DATA_DIR / "processed/zdc_filtered.csv"
# ZDC_DECK_FILE = DATA_DIR / "processed/zdc_deck.json"
# JAPAN_CITY_FILE = DATA_DIR / "raw/japan_city.pkl"


def exit_handler(finename):
    sc.api_call("chat.postMessage", channel="#experiment", text=finename + " is finished.")









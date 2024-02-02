import datetime
import os
import atexit
import configparser
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

conf = configparser.ConfigParser()
src_dir = Path(__file__).parent.resolve()
conf_file = src_dir / "config.cfg"
conf.read(conf_file)
SLACK_TOKEN = conf.get('Slack', 'token')
slack_client = slack.WebClient(token=SLACK_TOKEN)


##########
# EXPERIMENT_ENVIRONMENT = "local"
EXPERIMENT_ENVIRONMENT = "remote"


SCENARIO = 'usual'
# SCENARIO = 'fireworks'

VERSION = '20231226'  

##########

PROFILE_TABLE_NAME = f"experiment_profile_{EXPERIMENT_ENVIRONMENT}_{SCENARIO}_{VERSION}"
PROFILE_TABLE_NAME_SMALL_TARGET = f"experiment_profile_small_{EXPERIMENT_ENVIRONMENT}_{SCENARIO}_{VERSION}"
GPS_TABLE_RAW_NAME = f"experiment_traj_raw_{EXPERIMENT_ENVIRONMENT}_{SCENARIO}_{VERSION}"
GPS_TABLE_REGULARIZED_NAME = f"experiment_traj_reg_{EXPERIMENT_ENVIRONMENT}_{SCENARIO}_{VERSION}"

EXPERIMENT_PARAMETERS = {
        'EXPERIMENT_NAME': f"experiment_{EXPERIMENT_ENVIRONMENT}_{SCENARIO}_{VERSION}",
        'AOI' : [138.72, 34.9, 140.87, 36.28],
        'AOI_SMALL' : [139.8000000000000114,35.7083333333333357,139.8125000000000000,35.7250000000000014],
        'UNIT_TEMPORAL' : 10,  # in minutes
        'UNIT_TEMPORAL_TOPIC': 30,  # in minutes
        'UNIT_SPATIAL_METER' : 1000,
        'INPUT_DATASET' : "Interpolated",
        'PREDICTION_INPUT_LENGTH': 144,  # 144 for 24 hour in 10 mins
        'PREDICTION_OUTPUT_LENGTH': 6,  # 6 for 1 hour in 10 mins
        'num_topic_k': 10  # Number of topics
    }

# Scenarios
if SCENARIO == 'usual':
    EXPERIMENT_PARAMETERS['TIMESTART'] = "2012-07-25 00:00:00"
    EXPERIMENT_PARAMETERS['TIMEEND'] = "2012-07-26 23:59:59"
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START'] = "2012-07-25 08:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END'] = "2012-07-26 07:59:59"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START'] = "2012-07-26 08:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END'] = "2012-07-26 08:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_START'] = "2012-07-25 08:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_END'] = "2012-07-26 07:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_START'] = "2012-07-26 08:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_END'] = "2012-07-26 08:59:59"
    EXPERIMENT_PARAMETERS['TWEET_TABLE_NAME'] = "tweet_table_201207"

elif SCENARIO == 'fireworks':
    EXPERIMENT_PARAMETERS['TIMESTART'] = "2012-07-27 00:00:00"
    EXPERIMENT_PARAMETERS['TIMEEND'] = "2012-07-28 19:59:59"
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START'] = "2012-07-27 19:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END'] = "2012-07-28 18:59:59"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START'] = "2012-07-28 19:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END'] = "2012-07-28 19:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_START'] = "2012-07-27 19:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_END'] = "2012-07-28 18:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_START'] = "2012-07-28 19:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_END'] = "2012-07-28 19:59:59"
    EXPERIMENT_PARAMETERS['TWEET_TABLE_NAME'] = "tweet_table_201207"

elif SCENARIO == 'fireworks_noex':  
    EXPERIMENT_PARAMETERS['TIMESTART'] = "2012-07-27 00:00:00"
    EXPERIMENT_PARAMETERS['TIMEEND'] = "2012-07-28 19:59:59"
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START'] = "2012-07-27 18:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END'] = "2012-07-28 15:59:59"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START'] = "2012-07-28 18:00:00"
    EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END'] = "2012-07-28 18:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_START'] = "2012-07-27 19:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_END'] = "2012-07-28 18:59:59"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_START'] = "2012-07-28 19:00:00"
    EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_END'] = "2012-07-28 19:59:59"
    EXPERIMENT_PARAMETERS['TWEET_TABLE_NAME'] = "tweet_table_201207"

## Local
#####################################################################################
if EXPERIMENT_ENVIRONMENT == "local":
    EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE'] = 500
    EXPERIMENT_PARAMETERS['SAMPLE_SIZE'] = 500
    EXPERIMENT_PARAMETERS['EVALUATION_SAMPLE_SIZE'] = 500
    EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE'] = 100
    EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE'] = 500

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
    GPS_INTERPOLATED_FILTERED_EVALUATION_SMALL = DATA_DIR_PROCESSED / (
            "filtered_interpolated_evaluation_small_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")


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
    EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE'] = 5000000
    EXPERIMENT_PARAMETERS['SAMPLE_SIZE'] = 5000000
    EXPERIMENT_PARAMETERS['EVALUATION_SAMPLE_SIZE'] = 28000 # max 28000
    EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE'] = 1000
    EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE'] = 500000

    # EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE'] = 500
    # EXPERIMENT_PARAMETERS['SAMPLE_SIZE'] = 500
    # EXPERIMENT_PARAMETERS['EVALUATION_SAMPLE_SIZE'] = 500
    # EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE'] = 100
    # EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE'] = 500

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
    GPS_INTERPOLATED_FILTERED_EVALUATION_SMALL = DATA_DIR_PROCESSED / (
                "filtered_interpolated_evaluation_small_" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + ".csv")

    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    TEST_DB_TABLE_NAME = "gps_log_2"
#################################################################

# MODEL parameter prototyping notes
MODEL = {
    "dropout": 0.2,
    "num_layer": 2,
}

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

X_COORDINATE_FILE = TRAINING_DIR / "x_coordinate.npy"
Y_COORDINATE_FILE = TRAINING_DIR / "y_coordinate.npy"
X_GRID_FILE = TRAINING_DIR / "x_grid.npy"
Y_GRID_FILE = TRAINING_DIR / "y_grid.npy"
X_MODE_FILE = TRAINING_DIR / "x_mode.npy"
Y_MODE_FILE = TRAINING_DIR / "y_mode.npy"
X_TOPIC_FILE = TRAINING_DIR / "x_topic.npy"
Y_TOPIC_FILE = TRAINING_DIR / "y_topic.npy"

X_COORDINATE_EVALUATION_FILE = EVALUATION_DIR / "evaluation_x_coordinate.npy"
Y_COORDINATE_EVALUATION_FILE = EVALUATION_DIR / "evaluation_y_coordinate.npy"
X_GRID_EVALUATION_FILE = EVALUATION_DIR / "evaluation_x_grid.npy"
Y_GRID_EVALUATION_FILE = EVALUATION_DIR / "evaluation_y_grid.npy"
X_MODE_EVALUATION_FILE = EVALUATION_DIR / "evaluation_x_mode.npy"
Y_MODE_EVALUATION_FILE = EVALUATION_DIR / "evaluation_y_mode.npy"
X_TOPIC_EVALUATION_FILE = EVALUATION_DIR / "evaluation_x_topic.npy"
Y_TOPIC_EVALUATION_FILE = EVALUATION_DIR / "evaluation_y_topic.npy"

X_COORDINATE_EVALUATION_SMALL_FILE = EVALUATION_DIR / "evaluation_small_x_coordinate.npy"
Y_COORDINATE_EVALUATION_SMALL_FILE = EVALUATION_DIR / "evaluation_small_y_coordinate.npy"
X_GRID_EVALUATION_SMALL_FILE = EVALUATION_DIR / "evaluation_small_x_grid.npy"
Y_GRID_EVALUATION_SMALL_FILE = EVALUATION_DIR / "evaluation_small_y_grid.npy"
X_MODE_EVALUATION_SMALL_FILE = EVALUATION_DIR / "evaluation_small_x_mode.npy"
Y_MODE_EVALUATION_SMALL_FILE = EVALUATION_DIR / "evaluation_small_y_mode.npy"
X_TOPIC_EVALUATION_SMALL_FILE = EVALUATION_DIR / "evaluation_small_x_topic.npy"
Y_TOPIC_EVALUATION_SMALL_FILE = EVALUATION_DIR / "evaluation_small_y_topic.npy"

MODEL_FILE_TRANSFER_LSTM = TRAINING_DIR / "model_transfer_lstm.h5"
MODEL_FILE_LSTM = TRAINING_DIR / "model_lstm.h5"
MODEL_WEIGHT_FILE_LSTM = TRAINING_DIR / "model_weights_lstm.h5"
MODEL_FILE_LSTM_GRID = TRAINING_DIR / "model_lstm_grid.h5"
MODEL_WEIGHT_FILE_LSTM_GRID = TRAINING_DIR / "model_weights_lstm_grid.h5"

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

GEOJSON_FILE_EVALUATION_SMALL_OBSERVATION = EVALUATION_DIR / "trajectory_coordinate_small_observation.geojson"
GEOJSON_FILE_EVALUATION_SMALL_TRUE = EVALUATION_DIR / "trajectory_coordinate_small_true.geojson"
GEOJSON_FILE_EVALUATION_SMALL_PREDICTED_LSTM = EVALUATION_DIR / "trajectory_coordinate_small_predicted_lstm.geojson"
GEOJSON_FILE_EVALUATION_SMALL_OBSERVATION_GRID = EVALUATION_DIR / "trajectory_grid_small_observation.geojson"
GEOJSON_FILE_EVALUATION_SMALL_TRUE_GRID = EVALUATION_DIR / "trajectory_grid_small_true.geojson"
GEOJSON_FILE_EVALUATION_SMALL_PREDICTED_LSTM_GRID = EVALUATION_DIR / "trajectory_grid_small_predicted_lstm.geojson"

CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID = EVALUATION_DIR / "csv_trajectory_grid_observation.csv"
CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID = EVALUATION_DIR / "csv_trajectory_grid_true.csv"
CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID = EVALUATION_DIR / "csv_trajectory_grid_predicted.csv"

CSV_TRAJECTORY_FILE_EVALUATION_SMALL_OBSERVATION_GRID = EVALUATION_DIR / "csv_trajectory_grid_small_observation.csv"
CSV_TRAJECTORY_FILE_EVALUATION_SMALL_TRUE_GRID = EVALUATION_DIR / "csv_trajectory_grid_small_true.csv"
CSV_TRAJECTORY_FILE_EVALUATION_SMALL_PREDICTED_GRID = EVALUATION_DIR / "csv_trajectory_grid_small_predicted.csv"

CSV_TRAJECTORY_FILE_EVALUATION_RAW = EVALUATION_DIR / "csv_trajectory_raw.csv"

DICT_FILE = str(DATA_DIR_PROCESSED / "corpus.dict")
MM_CORPUS_FILE = str(DATA_DIR_PROCESSED / "corpus.mm")
TFIDF_FILE = str(DATA_DIR_PROCESSED / "tfidf.tfidf")
LSI_MODEL_FILE = str(DATA_DIR_PROCESSED / "Topic_LSI.model")
LDA_MODEL_FILE = str(DATA_DIR_PROCESSED / "Topic_LDA.model")
DOC2VEC_MODEL_FILE = str(DATA_DIR_PROCESSED / "Topic_Doc2Vec.model")

LSI_TOPIC_FILE = str(TRAINING_DIR / "Topic_Feature_LSI.npy")
LDA_TOPIC_FILE = str(TRAINING_DIR / "Topic_Feature_LDA.npy")
DOC2VEC_TOPIC_FILE = str(TRAINING_DIR / "Topic_Feature_Doc2Vec.npy")

LSI_TOPIC_EVALUATION_FILE = str(EVALUATION_DIR / "Topic_Feature_evaluation_LSI.npy")
LDA_TOPIC_EVALUATION_FILE = str(EVALUATION_DIR / "Topic_Feature_evaluation_LDA.npy")
DOC2VEC_TOPIC_EVALUATION_FILE = str(EVALUATION_DIR / "Topic_Feature_evaluation_Doc2Vec.npy")

LSI_TOPIC_SMALL_FILE = str(EVALUATION_DIR / "Topic_Feature_small_LSI.npy")
LDA_TOPIC_SMALL_FILE = str(EVALUATION_DIR / "Topic_Feature_small_LDA.npy")
DOC2VEC_TOPIC_SMALL_FILE = str(EVALUATION_DIR / "Topic_Feature_small_Doc2Vec.npy")


def exit_handler(finename, elapsed):
    slack_client.chat_postMessage(channel='#experiment', text=finename + " is finished in time: " + elapsed)


import csv
import atexit
import csv
import random
import matplotlib
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datetime import datetime

matplotlib.use('Agg')

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
start_time = datetime.now()

import settings as s
import utility_spatiotemporal_index
import prediction_6_gps_grid_mode_topic

np.random.seed(1)
random.seed(1)


def load_grid_mode_topic_dataset_evaluation(X_GRID_FILE, Y_GRID_FILE, X_MODE_FILE, Y_MODE_FILE, X_TOPIC_FILE, Y_TOPIC_FILE, le_grid, lb_mode, EXPERIMENT_PARAMETERS):
    EVALUATION_SAMPLE_SIZE = EXPERIMENT_PARAMETERS['EVALUATION_SAMPLE_SIZE']
    X_all = np.load(X_GRID_FILE)
    X_all = sample_for_evaluation(X_all, EVALUATION_SAMPLE_SIZE)
    y_all = np.load(Y_GRID_FILE)
    y_all = sample_for_evaluation(y_all, EVALUATION_SAMPLE_SIZE)

    X_all_shape = X_all.shape
    y_all_shape = y_all.shape
    X_all = X_all.reshape(X_all_shape[0] * X_all_shape[1], 1)
    y_all = y_all.reshape(y_all_shape[0] * y_all_shape[1], 1)
    X_all = le_grid.transform(X_all)
    y_all = le_grid.transform(y_all)
    X_all = X_all.reshape(X_all_shape[0], X_all_shape[1], 1)
    y_all = y_all.reshape(y_all_shape[0], y_all_shape[1], 1)

    X_mode_all = np.load(X_MODE_FILE)
    y_mode_all = np.load(Y_MODE_FILE)

    X_mode_all = sample_for_evaluation(X_mode_all, EVALUATION_SAMPLE_SIZE)
    y_mode_all = sample_for_evaluation(y_mode_all, EVALUATION_SAMPLE_SIZE)

    X_mode_all = X_mode_all.reshape(X_all.shape[0] * X_all.shape[1], 1)
    y_mode_all = y_mode_all.reshape(y_all.shape[0] * y_all.shape[1], 1)
    num_mode = len(lb_mode.classes_)
    X_mode_all = lb_mode.transform(X_mode_all)
    y_mode_all = lb_mode.transform(y_mode_all)
    X_mode_all = X_mode_all.reshape(X_all.shape[0], X_all.shape[1], num_mode)
    y_mode_all = y_mode_all.reshape(y_all.shape[0], y_all.shape[1], num_mode)

    X_topic_all = np.load(X_TOPIC_FILE)
    y_topic_all = np.load(Y_TOPIC_FILE)

    X_topic_all = sample_for_evaluation(X_topic_all, EVALUATION_SAMPLE_SIZE)
    y_topic_all = sample_for_evaluation(y_topic_all, EVALUATION_SAMPLE_SIZE)

    # X_topic_all = X_topic_all.reshape(X_all.shape[0] * X_all.shape[1], 1)
    # y_topic_all = y_topic_all.reshape(y_all.shape[0] * y_all.shape[1], 1)
    # X_topic_all = X_topic_all.reshape(X_all.shape[0], X_all.shape[1], num_mode)
    # y_topic_all = y_topic_all.reshape(y_all.shape[0], y_all.shape[1], num_mode)

    return X_all, y_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all


def sample_for_evaluation(input_array, EVALUATION_SAMPLE_SIZE):
    output_array = input_array[:EVALUATION_SAMPLE_SIZE]
    return output_array


def convert_spatial_index_array_to_coordinate_array(input_array, le_grid, Multistep=False):
    if Multistep:
        X_shape = input_array.shape
        # print(X_shape) # (18, 12, 1)
        input_array = input_array.reshape(X_shape[0] * X_shape[1], 1)
        input_array = le_grid.inverse_transform(input_array)
        input_array = input_array.reshape(X_shape)
        x_array = np.apply_along_axis(np.vectorize(utility_spatiotemporal_index.convert_spatial_index_to_longitude), 0, input_array)
        y_array = np.apply_along_axis(np.vectorize(utility_spatiotemporal_index.convert_spatial_index_to_latitude), 0, input_array)
        latlon_array = np.concatenate((y_array, x_array), axis=2)
    else:
        X_shape = input_array.shape
        # print(X_shape)  # (18, 12, 1)
        # print(input_array)
        input_array = le_grid.inverse_transform(input_array)
        # print(input_array)
        x_array = np.apply_along_axis(np.vectorize(utility_spatiotemporal_index.convert_spatial_index_to_longitude), 0, input_array)
        y_array = np.apply_along_axis(np.vectorize(utility_spatiotemporal_index.convert_spatial_index_to_latitude), 0, input_array)
        # latlon_array = np.concatenate((y_array, x_array), axis=1)
        latlon_array = np.column_stack((y_array, x_array))
        # print(latlon_array)
    return latlon_array


def create_csv_trajectory_observation(X, csv_file, temporal_index, EXPERIMENT_PARAMETERS):
    logger.info("Saving trajectory to CSV %s" % csv_file)
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        X_shape = X.shape
        # print(X_shape) # (50, 144, 2)
        X_list = X.tolist()
        # print(X_list)
        for i in range(len(X_list)):
            for j in range(len(X_list[0])):
                id = i
                t = temporal_index[j][4].strftime('%Y-%m-%d %H:%M:%S')
                x = X_list[i][j][1]
                y = X_list[i][j][0]
                writer.writerow([id, t, x, y])
    return True


def create_csv_trajectory_prediction(X, y, csv_file, temporal_index, EXPERIMENT_PARAMETERS):
    logger.info("Saving trajectory to CSV %s" % csv_file)
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        X_shape = X.shape
        # print(X_shape) # (50, 144, 2)
        y_shape = y.shape
        # print(y_shape) # (50, 6, 2)
        X_list = X.tolist()
        y_list = y.tolist()
        # print(len(temporal_index)) # 150
        # print(X_list)
        for i in range(len(X_list)):
            id = i
            t_1 = temporal_index[X_shape[1] - 1][4].strftime('%Y-%m-%d %H:%M:%S')
            x_1 = X_list[i][-1][1]
            y_1 = X_list[i][-1][0]
            writer.writerow([id, t_1, x_1, y_1])
            for j in range(len(y_list[0])):
                t = temporal_index[X_shape[1] + j][4].strftime('%Y-%m-%d %H:%M:%S')
                x = y_list[i][j][1]
                y = y_list[i][j][0]
                writer.writerow([id, t, x, y])
    return True

def select_moving_array(array_x, array_y, array_predicted):
    mask = array_x[:, -1, 0] != array_y[:, 0, 0]
    array_x = array_x[mask[:], :, :]
    array_y = array_y[mask[:], :, :]
    array_predicted = array_predicted[mask[:], :, :]
    mask = array_x[:, -1, 1] != array_y[:, 0, 1]
    array_x = array_x[mask[:], :, :]
    array_y = array_y[mask[:], :, :]
    array_predicted = array_predicted[mask[:], :, :]
    return array_x, array_y, array_predicted


def select_moving_array_all_features(array_x, array_y, array_predicted, X_all, y_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all):
    mask = array_x[:, -1, 0] != array_y[:, 0, 0]
    array_x = array_x[mask[:], :, :]
    array_y = array_y[mask[:], :, :]
    array_predicted = array_predicted[mask[:], :, :]
    X_all = X_all[mask[:], :, :]
    y_all = y_all[mask[:], :, :]
    X_mode_all = X_mode_all[mask[:], :, :]
    y_mode_all = y_mode_all[mask[:], :, :]
    X_topic_all = X_topic_all[mask[:], :, :]
    y_topic_all = y_topic_all[mask[:], :, :]
    mask = array_x[:, -1, 1] != array_y[:, 0, 1]
    array_x = array_x[mask[:], :, :]
    array_y = array_y[mask[:], :, :]
    array_predicted = array_predicted[mask[:], :, :]
    X_all = X_all[mask[:], :, :]
    y_all = y_all[mask[:], :, :]
    X_mode_all = X_mode_all[mask[:], :, :]
    y_mode_all = y_mode_all[mask[:], :, :]
    X_topic_all = X_topic_all[mask[:], :, :]
    y_topic_all = y_topic_all[mask[:], :, :]
    return array_x, array_y, array_predicted, X_all, y_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all


if __name__ == '__main__':
    slack_client = s.slack_client
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR
    X_GRID_EVALUATION_FILE = s.X_GRID_EVALUATION_SMALL_FILE
    Y_GRID_EVALUATION_FILE = s.Y_GRID_EVALUATION_SMALL_FILE
    X_MODE_EVALUATION_FILE = s.X_MODE_EVALUATION_SMALL_FILE
    Y_MODE_EVALUATION_FILE = s.Y_MODE_EVALUATION_SMALL_FILE
    X_TOPIC_EVALUATION_FILE = s.X_TOPIC_EVALUATION_SMALL_FILE
    Y_TOPIC_EVALUATION_FILE = s.Y_TOPIC_EVALUATION_SMALL_FILE
    LE_GRID_CLASSES_FILE = s.LE_GRID_CLASSES_FILE
    # Y_FILE_PREDICTED_LSTM = s.Y_FILE_PREDICTED_LSTM
    # Y_FILE_PREDICTED_VELOCITY = s.Y_FILE_PREDICTED_VELOCITY
    LE_GRID_CLASSES_FILE = s.LE_GRID_CLASSES_FILE
    LB_MODE_CLASSES_FILE = s.LB_MODE_CLASSES_FILE

    MODEL_FILE_LSTM_GRID = s.MODEL_FILE_LSTM_GRID
    MODEL_WEIGHT_FILE_LSTM_GRID = s.MODEL_WEIGHT_FILE_LSTM_GRID

    GEOJSON_FILE_EVALUATION_OBSERVATION_GRID = s.GEOJSON_FILE_EVALUATION_SMALL_OBSERVATION_GRID
    GEOJSON_FILE_EVALUATION_TRUE_GRID = s.GEOJSON_FILE_EVALUATION_SMALL_TRUE_GRID
    GEOJSON_FILE_EVALUATION_PREDICTED_LSTM_GRID = s.GEOJSON_FILE_EVALUATION_SMALL_PREDICTED_LSTM_GRID

    CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_SMALL_OBSERVATION_GRID
    CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_SMALL_TRUE_GRID
    CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_SMALL_PREDICTED_GRID

    le_grid = LabelEncoder()
    le_grid.classes_ = np.load(LE_GRID_CLASSES_FILE)
    lb_mode = LabelBinarizer()
    lb_mode.classes_ = np.load(LB_MODE_CLASSES_FILE)

    temporal_index = utility_spatiotemporal_index.define_temporal_index_evaluation(EXPERIMENT_PARAMETERS)

    X_all, y_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all = load_grid_mode_topic_dataset_evaluation(X_GRID_EVALUATION_FILE, Y_GRID_EVALUATION_FILE, X_MODE_EVALUATION_FILE, Y_MODE_EVALUATION_FILE, X_TOPIC_EVALUATION_FILE, Y_TOPIC_EVALUATION_FILE, le_grid, lb_mode, EXPERIMENT_PARAMETERS)

    # Load model
    lstm_model = load_model(MODEL_FILE_LSTM_GRID)

    y_predicted_sequence, error_meter_series = prediction_6_gps_grid_mode_topic.prediction_multiple_steps_lstm_grid_mode_topic(lstm_model, X_all, y_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all, le_grid, EXPERIMENT_PARAMETERS)

    # print(error_meter_series.shape)
    error_meter_series_sort_id = np.argsort(error_meter_series)
    # print(error_meter_series[error_meter_series_sort_id])

    X_test_latlon = prediction_6_gps_grid_mode_topic.convert_spatial_index_array_to_coordinate_array(X_all, le_grid, Multistep=True)
    y_test_latlon = prediction_6_gps_grid_mode_topic.convert_spatial_index_array_to_coordinate_array(y_all, le_grid, Multistep=True)

    X_test_latlon = X_test_latlon[error_meter_series_sort_id]
    y_test_latlon = y_test_latlon[error_meter_series_sort_id]
    y_predicted_sequence = y_predicted_sequence[error_meter_series_sort_id]

    X_test_latlon, y_test_latlon, y_predicted_sequence = select_moving_array(X_test_latlon, y_test_latlon, y_predicted_sequence)

    create_csv_trajectory_observation(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID, temporal_index, EXPERIMENT_PARAMETERS)
    create_csv_trajectory_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], y_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID, temporal_index, EXPERIMENT_PARAMETERS)
    create_csv_trajectory_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], y_predicted_sequence[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID, temporal_index, EXPERIMENT_PARAMETERS)

    EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE'] = min(EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE'], X_test_latlon.shape[0])
    prediction_6_gps_grid_mode_topic.create_geojson_line_observation(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_EVALUATION_OBSERVATION_GRID, EXPERIMENT_PARAMETERS)
    prediction_6_gps_grid_mode_topic.create_geojson_line_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_EVALUATION_TRUE_GRID, EXPERIMENT_PARAMETERS)
    prediction_6_gps_grid_mode_topic.create_geojson_line_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_predicted_sequence[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_EVALUATION_PREDICTED_LSTM_GRID, EXPERIMENT_PARAMETERS)


    elapsed = str((datetime.now() - start_time))
    logger.info(f"Finished in: {elapsed}")
    atexit.register(s.exit_handler, __file__, elapsed)
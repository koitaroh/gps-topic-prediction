import requests
import json
import csv
import os
import time
import atexit
import random
import sqlalchemy
import geojson
from geojson import LineString, FeatureCollection, Feature
import pandas
import pickle
import numpy as np
import tensorflow as tf
import h5py
from geopy.distance import vincenty
from datetime import datetime, timedelta
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
import math
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM, SimpleRNN, GRU, Concatenate, Dense, Dropout, Activation, Embedding, TimeDistributed, Flatten
from keras.datasets import imdb
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

import settings as s
import load_dataset
import jpgrid
import utility_io
import utility_spatiotemporal_index
import prediction_gps_grid_mode_topic

np.random.seed(1)
random.seed(1)


def load_grid_mode_topic_dataset_evaluation(X_GRID_FILE, Y_GRID_FILE, X_MODE_FILE, Y_MODE_FILE, X_TOPIC_FILE, Y_TOPIC_FILE, le_grid, lb_mode, EXPERIMENT_PARAMETERS):
    X_all = np.load(X_GRID_FILE)
    y_all = np.load(Y_GRID_FILE)
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
    X_mode_all = X_mode_all.reshape(X_all.shape[0] * X_all.shape[1], 1)
    y_mode_all = y_mode_all.reshape(y_all.shape[0] * y_all.shape[1], 1)
    num_mode = len(lb_mode.classes_)
    X_mode_all = lb_mode.transform(X_mode_all)
    y_mode_all = lb_mode.transform(y_mode_all)
    X_mode_all = X_mode_all.reshape(X_all.shape[0], X_all.shape[1], num_mode)
    y_mode_all = y_mode_all.reshape(y_all.shape[0], y_all.shape[1], num_mode)

    X_topic_all = np.load(X_TOPIC_FILE)
    y_topic_all = np.load(Y_TOPIC_FILE)
    # X_topic_all = X_topic_all.reshape(X_all.shape[0] * X_all.shape[1], 1)
    # y_topic_all = y_topic_all.reshape(y_all.shape[0] * y_all.shape[1], 1)
    # X_topic_all = X_topic_all.reshape(X_all.shape[0], X_all.shape[1], num_mode)
    # y_topic_all = y_topic_all.reshape(y_all.shape[0], y_all.shape[1], num_mode)


    return X_all, y_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all


# def convert_spatial_index_to_raw_coordinates(EXPERIMENT_PARAMETERS, spatial_index, spatial_index_number):
#     spatial_index_number = int(spatial_index_number)
#     y_index = spatial_index_number // spatial_index[3]
#     x_index = spatial_index_number - (y_index * spatial_index[3])
#     print(x_index)
#     print(y_index)
#     x = EXPERIMENT_PARAMETERS["AOI"][0] + (spatial_index[0] * x_index) + (spatial_index[0] / 2)
#     y = EXPERIMENT_PARAMETERS["AOI"][1] + (spatial_index[1] * y_index) + (spatial_index[1] / 2)
#     return x, y
#
#
# def convert_spatial_index_to_latitude(spatial_index_number, EXPERIMENT_PARAMETERS, spatial_index):
#     y_index = spatial_index_number // spatial_index[3]
#     y = EXPERIMENT_PARAMETERS["AOI"][1] + (spatial_index[1] * y_index) + (spatial_index[1] / 2)
#     return y
#
#
# def convert_spatial_index_to_longitude(spatial_index_number, EXPERIMENT_PARAMETERS, spatial_index):
#     y_index = spatial_index_number // spatial_index[3]
#     x_index = spatial_index_number - (y_index * spatial_index[3])
#     x = EXPERIMENT_PARAMETERS["AOI"][0] + (spatial_index[0] * x_index) + (spatial_index[0] / 2)
#     return x
#
#
# def convert_spatial_index_array_to_coordinate_array(input_array, EXPERIMENT_PARAMETERS, spatial_index, Multistep=False):
#     if Multistep:
#         X_shape = input_array.shape
#         # print(X_shape) # (18, 12, 1)
#         x_array = np.apply_along_axis(convert_spatial_index_to_longitude, 0, input_array, EXPERIMENT_PARAMETERS, spatial_index)
#         y_array = np.apply_along_axis(convert_spatial_index_to_latitude, 0, input_array, EXPERIMENT_PARAMETERS, spatial_index)
#         latlon_array = np.concatenate((y_array, x_array), axis=2)
#
#     else:
#         x_array = np.apply_along_axis(convert_spatial_index_to_longitude, 0, input_array, EXPERIMENT_PARAMETERS,
#                                       spatial_index)
#         y_array = np.apply_along_axis(convert_spatial_index_to_latitude, 0, input_array, EXPERIMENT_PARAMETERS,
#                                       spatial_index)
#         latlon_array = np.concatenate((y_array, x_array), axis=1)
#     return latlon_array


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
        latlon_array = np.concatenate((y_array, x_array), axis=1)
        # print(latlon_array)
    return latlon_array


# def prediction_multiple_steps(model, X, y, X_mode, y_mode, X_topic, y_topic, spatial_index, EXPERIMENT_PARAMETERS):
#     PREDICTION_OUTPUT_LENGTH = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
#     X_middle_step = X
#     X_mode_middle_step = X_mode
#     X_topic_middle_step = X_topic
#     for i in tqdm(range(PREDICTION_OUTPUT_LENGTH)):
#         if i == 0:
#             y_predicted = model.predict([X_middle_step, X_mode_middle_step, X_topic_middle_step])
#             y_true = y[:, i, :]
#             y_mode_true = y_mode[:, i, :]
#             y_topic_true = y_topic[:, i, :]
#             # y_predicted_s_indices = np.argmax(y_predicted, axis=1) # For selecting max
#             y_predicted_s_indices = np.array([*map(lambda p: np.random.choice(max_s_index, p=p), y_predicted)]) # For selecting as random sample
#             y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
#             y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, EXPERIMENT_PARAMETERS, spatial_index)
#             y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, EXPERIMENT_PARAMETERS, spatial_index)
#             rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
#             logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
#             X_middle_step = X_middle_step[:, 1:, :]
#             X_mode_middle_step = X_mode_middle_step[:, 1:, :]
#             X_topic_middle_step = X_topic_middle_step[:, 1:, :]
#             y_predicted_s_indices = y_predicted_s_indices.reshape(y_predicted_s_indices.shape[0], 1, y_predicted_s_indices.shape[1])
#             y_mode_true = y_mode_true.reshape(y_mode_true.shape[0], 1, y_mode_true.shape[1])
#             y_topic_true = y_topic_true.reshape(y_topic_true.shape[0], 1, y_topic_true.shape[1])
#             X_middle_step = np.concatenate((X_middle_step, y_predicted_s_indices), axis=1)
#             X_mode_middle_step = np.concatenate((X_mode_middle_step, y_mode_true), axis=1)
#             X_topic_middle_step = np.concatenate((X_topic_middle_step, y_topic_true), axis=1)
#             y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
#             y_predicted_sequence = y_predicted_latlon
#
#         elif i == PREDICTION_OUTPUT_LENGTH:
#             y_true = y[:, i, :]
#             y_mode_true = y_mode[:, i, :]
#             y_topic_true = y_topic[:, i, :]
#             y_predicted = model.predict([X_middle_step, X_mode_middle_step, X_topic_middle_step])
#             # y_predicted_s_indices = np.argmax(y_predicted, axis=1) # For selecting max
#             y_predicted_s_indices = np.array([*map(lambda p: np.random.choice(max_s_index, p=p), y_predicted)])  # For selecting as random sample
#             y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
#             y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, EXPERIMENT_PARAMETERS, spatial_index)
#             y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, EXPERIMENT_PARAMETERS, spatial_index)
#             rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
#             logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
#             y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
#             y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)
#
#         else:
#             y_predicted = model.predict([X_middle_step, X_mode_middle_step, X_topic_middle_step])
#             y_true = y[:, i, :]
#             y_mode_true = y_mode[:, i, :]
#             y_topic_true = y_topic[:, i, :]
#             # y_predicted_s_indices = np.argmax(y_predicted, axis=1) # For selecting max
#             y_predicted_s_indices = np.array([*map(lambda p: np.random.choice(max_s_index, p=p), y_predicted)])  # For selecting as random sample
#             y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
#             y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, EXPERIMENT_PARAMETERS, spatial_index)
#             y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, EXPERIMENT_PARAMETERS, spatial_index)
#             rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
#             logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
#             X_middle_step = X_middle_step[:, 1:, :]
#             X_mode_middle_step = X_mode_middle_step[:, 1:, :]
#             X_topic_middle_step = X_topic_middle_step[:, 1:, :]
#             y_predicted_s_indices = y_predicted_s_indices.reshape(y_predicted_s_indices.shape[0], 1, y_predicted_s_indices.shape[1])
#             y_mode_true = y_mode_true.reshape(y_mode_true.shape[0], 1, y_mode_true.shape[1])
#             y_topic_true = y_topic_true.reshape(y_topic_true.shape[0], 1, y_topic_true.shape[1])
#             X_middle_step = np.concatenate((X_middle_step, y_predicted_s_indices), axis=1)
#             X_mode_middle_step = np.concatenate((X_mode_middle_step, y_mode_true), axis=1)
#             X_topic_middle_step = np.concatenate((X_topic_middle_step, y_topic_true), axis=1)
#             y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
#             y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)
#     # print(y_predicted_sequence)
#     # print(y_predicted_sequence.shape)
#     return y_predicted_sequence


# def calculate_rmse_in_meters(input_array):
#     origin = (input_array[0], input_array[1])
#     destination = (input_array[2], input_array[3])
#     distance = vincenty(origin, destination).meters
#     return distance
#
#
# def calculate_rmse_on_array(y_predicted, y_test):
#     y_join = np.concatenate((y_predicted, y_test), axis=1)
#     # print(y_join)
#     # print(y_join.shape)
#     error_meter = np.apply_along_axis(calculate_rmse_in_meters, axis=1, arr=y_join)
#     # print(error_meter.shape)
#     rmse_meter_mean = error_meter.mean()
#     # logger.info("RMSE in meters: %s" % rmse_meter_mean)
#     return rmse_meter_mean


# def create_geojson_line_prediction(X, y, outfile, EXPERIMENT_PARAMETERS):
#     logger.info("Creating GeoJSON file")
#     predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
#     with open(outfile, 'w', encoding="utf-8") as geojson_file:
#         X_shape = X.shape
#         if predict_length > 1:
#             y_length = y.shape[1]
#         else:
#             y_length = 1
#         my_feature_list = []
#         for i in range(X_shape[0]):
#             properties = {"new_uid": i}
#             line = []
#             lat, lon = X[i, -1, -2], X[i, -1, -1]
#             line.append((lon, lat))
#             if y_length == 1:
#                 lat, lon = np.asscalar(y[i, -2]), np.asscalar(y[i, -1])
#                 line.append((lon, lat))
#             else:
#                 for j in range(y_length):
#                     lat, lon = np.asscalar(y[i, j, -2]), np.asscalar(y[i, j, -1])
#                     line.append((lon, lat))
#             my_line = LineString(line)
#             my_feature = Feature(geometry=my_line, properties=properties)
#             my_feature_list.append(my_feature)
#         my_feature_collection = FeatureCollection(my_feature_list)
#         dump = geojson.dumps(my_feature_collection, sort_keys=True)
#         geojson_file.write(dump)
#     return None
#
#
# def create_geojson_line_observation(X, outfile, EXPERIMENT_PARAMETERS):
#     logger.info("Creating GeoJSON file")
#     predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
#     with open(outfile, 'w', encoding="utf-8") as geojson_file:
#         X_shape = X.shape
#         my_feature_list = []
#         for i in range(X_shape[0]):
#             properties = {"new_uid": i}
#             line = []
#             for j in range(X_shape[1]):
#                 lat, lon = X[i, j, -2], X[i, j, -1]
#                 line.append((lon, lat))
#             my_line = LineString(line)
#             my_feature = Feature(geometry=my_line, properties=properties)
#             my_feature_list.append(my_feature)
#         my_feature_collection = FeatureCollection(my_feature_list)
#         dump = geojson.dumps(my_feature_collection, sort_keys=True)
#         geojson_file.write(dump)
#     return None


def create_csv_trajectory_observation(X, csv_file, temporal_index, EXPERIMENT_PARAMETERS):
    logger.info("Saving trajectory to CSV %s" % csv_file)
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        X_shape = X.shape
        # print(X_shape) # (50, 144, 2)
        X_list = X.tolist()
        # print(X_list)
        for i in tqdm(range(len(X_list))):
            for j in tqdm(range(len(X_list[0]))):
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
        for i in tqdm(range(len(X_list))):
            id = i
            t_1 = temporal_index[X_shape[1] - 1][4].strftime('%Y-%m-%d %H:%M:%S')
            x_1 = X_list[i][-1][1]
            y_1 = X_list[i][-1][0]
            writer.writerow([id, t_1, x_1, y_1])
            for j in tqdm(range(len(y_list[0]))):
                t = temporal_index[X_shape[1] + j][4].strftime('%Y-%m-%d %H:%M:%S')
                x = y_list[i][j][1]
                y = y_list[i][j][0]
                writer.writerow([id, t, x, y])
    return True


if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR
    X_MODE_EVALUATION_FILE = s.X_MODE_EVALUATION_FILE
    X_GRID_EVALUATION_FILE = s.X_GRID_EVALUATION_FILE
    Y_GRID_EVALUATION_FILE = s.Y_GRID_EVALUATION_FILE
    Y_MODE_EVALUATION_FILE = s.Y_MODE_EVALUATION_FILE
    X_TOPIC_EVALUATION_FILE = s.X_TOPIC_EVALUATION_FILE
    Y_TOPIC_EVALUATION_FILE = s.Y_TOPIC_EVALUATION_FILE
    LE_GRID_CLASSES_FILE = s.LE_GRID_CLASSES_FILE
    X_FILE = s.X_FILE
    Y_FILE = s.Y_FILE
    Y_FILE_PREDICTED_LSTM = s.Y_FILE_PREDICTED_LSTM
    Y_FILE_PREDICTED_VELOCITY = s.Y_FILE_PREDICTED_VELOCITY
    LE_GRID_CLASSES_FILE = s.LE_GRID_CLASSES_FILE
    LB_MODE_CLASSES_FILE = s.LB_MODE_CLASSES_FILE

    MODEL_FILE_LSTM_GRID = s.MODEL_FILE_LSTM_GRID
    MODEL_WEIGHT_FILE_LSTM_GRID = s.MODEL_WEIGHT_FILE_LSTM_GRID

    GEOJSON_FILE_EVALUATION_OBSERVATION_GRID = s.GEOJSON_FILE_EVALUATION_OBSERVATION_GRID
    GEOJSON_FILE_EVALUATION_TRUE_GRID = s.GEOJSON_FILE_EVALUATION_TRUE_GRID
    GEOJSON_FILE_EVALUATION_PREDICTED_LSTM_GRID = s.GEOJSON_FILE_EVALUATION_PREDICTED_LSTM_GRID

    CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID
    CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID
    CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID

    le_grid = LabelEncoder()
    le_grid.classes_ = np.load(LE_GRID_CLASSES_FILE)
    lb_mode = LabelBinarizer()
    lb_mode.classes_ = np.load(LB_MODE_CLASSES_FILE)

    temporal_index = utility_spatiotemporal_index.define_temporal_index_evaluation(EXPERIMENT_PARAMETERS)

    X_all, y_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all = load_grid_mode_topic_dataset_evaluation(X_GRID_EVALUATION_FILE, Y_GRID_EVALUATION_FILE, X_MODE_EVALUATION_FILE, Y_MODE_EVALUATION_FILE, X_TOPIC_EVALUATION_FILE, Y_TOPIC_EVALUATION_FILE, le_grid, lb_mode, EXPERIMENT_PARAMETERS)

    # Load model
    lstm_model = load_model(MODEL_FILE_LSTM_GRID)

    y_predicted_sequence = prediction_gps_grid_mode_topic.prediction_multiple_steps(lstm_model, X_all, y_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all, le_grid, EXPERIMENT_PARAMETERS)

    X_test_latlon = prediction_gps_grid_mode_topic.convert_spatial_index_array_to_coordinate_array(X_all, le_grid, Multistep=True)
    y_test_latlon = prediction_gps_grid_mode_topic.convert_spatial_index_array_to_coordinate_array(y_all, le_grid, Multistep=True)

    prediction_gps_grid_mode_topic.create_geojson_line_observation(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_EVALUATION_OBSERVATION_GRID, EXPERIMENT_PARAMETERS)
    prediction_gps_grid_mode_topic.create_geojson_line_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_EVALUATION_TRUE_GRID, EXPERIMENT_PARAMETERS)
    prediction_gps_grid_mode_topic.create_geojson_line_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_predicted_sequence[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_EVALUATION_PREDICTED_LSTM_GRID, EXPERIMENT_PARAMETERS)

    # print(temporal_index)
    create_csv_trajectory_observation(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID, temporal_index, EXPERIMENT_PARAMETERS)
    create_csv_trajectory_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], y_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID, temporal_index, EXPERIMENT_PARAMETERS)
    create_csv_trajectory_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], y_predicted_sequence[0:EXPERIMENT_PARAMETERS['VISUALIZATION_CSV_SAMPLE_SIZE']], CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID, temporal_index, EXPERIMENT_PARAMETERS)

    atexit.register(s.exit_handler, __file__) # Make notification
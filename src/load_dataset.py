import requests
import json
import csv
import os
import time
import random
import sqlalchemy
import pandas
import matplotlib.pyplot as plt
import pickle
import numpy as np
import datetime
from datetime import datetime, timedelta
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import settings as s

np.random.seed(7)
random.seed(7)


# Logging ver. 2016-07-12
from logging import handlers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('log.log', maxBytes=1000000, backupCount=3)  # file handler
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # console handler
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Initializing %s', __name__)

# Load filetered / regularized CSV for prediction model


def load_csv_to_dataframe(CSV_FILE, regularized=False):
    logger.info("Loading CSV %s to dataframe" % CSV_FILE)

    if (regularized):
        headers = ['uid', 't_index', 'iso_year', 'iso_week_number', 'iso_weekday', 'timestamp', 'y', 'x']
        # dtype = {'uid': 'str', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}
        dtype = {'uid': 'str', 't_index': 'int64', 'iso_year': 'int64', 'iso_week_number': 'int64', 'iso_weekday': 'int64', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}

        parse_dates = ['timestamp']
        df_csv = pandas.read_csv(filepath_or_buffer=CSV_FILE, header=None, names=headers,
                                 dtype=dtype, parse_dates=parse_dates, skiprows=1,
                                 usecols=[0, 1, 2, 3, 4, 5, 6, 7], error_bad_lines=False, warn_bad_lines=True)
    else:

        headers = ['uid', 'timestamp', 'y', 'x']
        dtype = {'uid': 'str', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}
        parse_dates = ['timestamp']
        df_csv = pandas.read_csv(filepath_or_buffer=CSV_FILE, header=None, names=headers,
                                 dtype=dtype, parse_dates=parse_dates,
                                 usecols=[0,1,2,3], error_bad_lines=False, warn_bad_lines=True)
    df_csv = df_csv.sort_values(by=['uid', 'timestamp'])
    return df_csv


def load_coordinates_numpy_input_file(X_all, y_all):
    X_all = np.load(X_all)
    y_all = np.load(y_all)
    # print(X_all)
    # print(y_all)

    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    x_shape = X_all.shape
    y_shape = y_all.shape
    X_all = X_all.reshape(x_shape[0] * x_shape[1], x_shape[2])
    y_all = y_all.reshape(y_shape[0] * y_shape[1], y_shape[2])
    X_y_all = np.concatenate((X_all, y_all), axis=0)

    # y_one_step = y_all[:,0,:]
    scaler.fit(X_y_all)
    X_all = X_all.reshape(x_shape[0], x_shape[1], x_shape[2])
    y_all = y_all.reshape(y_shape[0], y_shape[1], y_shape[2])

    return X_all, y_all, scaler



def devide_sample(X_all, y_all, scaler, EXPERIMENT_PARAMETERS):
    # logger.debug(len(X_all))

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, test_size=0.2, random_state=7)

    #
    # p = np.random.permutation(len(X_all))
    # X_all = X_all[p]
    # y_all = y_all[p]
    #
    # num_train = int(len(X_all) * 0.8)
    # X_train = X_all[0:num_train]
    # X_test = X_all[num_train:]
    # y_train = y_all[0:num_train]
    # y_test = y_all[num_train:]

    X_train = scale_transform_sample(X_train, scaler, Multistep=True)
    X_test = scale_transform_sample(X_test, scaler, Multistep=True)
    y_train = scale_transform_sample(y_train, scaler, Multistep=True)
    y_test = scale_transform_sample(y_test, scaler, Multistep=True)

    return (X_train, y_train), (X_test, y_test)


def create_full_training_sample(X_train_original, y_train_original):
    PREDICTION_OUTPUT_LENGTH = y_train_original.shape[1]
    for i in range(PREDICTION_OUTPUT_LENGTH):
        if i == 0:
            X_train_all = X_train_original
            y_train_all = y_train_original[:, i, :]
        else:
            X = X_train_original[:, i:, :]
            X = np.concatenate((X, y_train_original[:, 0:i, :]), axis=1)
            y = y_train_original[:, i, :]
            X_train_all = np.concatenate((X_train_all, X), axis=0)
            y_train_all = np.concatenate((y_train_all, y), axis=0)
    return X_train_all, y_train_all


def scale_transform_sample(input_array, scaler, Multistep=False):
    if Multistep:
        X_shape = input_array.shape
        # print(X_shape)
        input_array = input_array.reshape(X_shape[0] * X_shape[1], X_shape[2])
        output_array = scaler.transform(input_array)
        output_array = output_array.reshape(X_shape[0], X_shape[1], X_shape[2])
    else:
        output_array = scaler.transform(input_array)
    return output_array


def inverse_scale_transform_sample(input_array, scaler, Multistep=False):
    if Multistep:
        X_shape = input_array.shape
        # print(X_shape)
        input_array = input_array.reshape(X_shape[0] * X_shape[1], X_shape[2])
        output_array = scaler.inverse_transform(input_array)
        output_array = output_array.reshape(X_shape[0], X_shape[1], X_shape[2])
    else:
        output_array = scaler.inverse_transform(input_array)
    return output_array


# def load_coordinates_dataset(X_COORDINATE_FILE, Y_COORDINATE_FILE, EXPERIMENT_PARAMETERS):
#     X_all, y_all, scaler = load_coordinates_numpy_input_file(X_COORDINATE_FILE, Y_COORDINATE_FILE)
#     (X_train, y_train), (X_test, y_test) = devide_sample(X_all, y_all, scaler, EXPERIMENT_PARAMETERS)
#     X_train, y_train = create_full_training_sample(X_train, y_train)
#     return (X_train, y_train), (X_test, y_test), (scaler)



if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS

    X_COORDINATE_FILE = s.X_COORDINATE_FILE
    Y_COORDINATE_FILE = s.Y_COORDINATE_FILE
    X_GRID_FILE = s.X_GRID_FILE
    Y_GRID_FILE = s.Y_GRID_FILE

    spatial_index_size = 30492

    # X_all, y_all, scaler = load_coordinates_numpy_input_file(X_COORDINATE_FILE, Y_COORDINATE_FILE)
    # (X_train, y_train), (X_test, y_test) = devide_sample(X_all, y_all, scaler, EXPERIMENT_PARAMETERS)
    # X_train, y_train = create_full_training_sample(X_train, y_train)

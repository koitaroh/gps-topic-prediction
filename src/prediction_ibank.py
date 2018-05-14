import requests
import json
import csv
import os
import time
import random
import sqlalchemy
import geojson
from geojson import LineString, FeatureCollection, Feature
import pandas
import matplotlib.pyplot as plt
import pickle
import numpy
import h5py
from geopy.distance import vincenty
import datetime
from datetime import datetime, timedelta
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
import math
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from project.references import settings as s
import load_dataset
import predict

numpy.random.seed(7)
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



if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    TRANSFER_CSV_FILTERED = s.TRANSFER_CSV_FILTERED
    FIGURE_DIR = s.FIGURE_DIR


    # MSE_FIGURE_FILE = s.MSE_FIGURE_FILE
    # MAE_FIGURE_FILE = s.MAE_FIGURE_FILE
    # MAPE_FIGURE_FILE = s.MAPE_FIGURE_FILE

    X_FILE = s.X_FILE
    Y_FILE = s.Y_FILE
    Y_FILE_PREDICTED_LSTM = s.Y_FILE_PREDICTED_LSTM
    Y_FILE_PREDICTED_VELOCITY = s.Y_FILE_PREDICTED_VELOCITY

    MODEL_FILE_LSTM = s.MODEL_FILE_LSTM
    MODEL_WEIGHT_FILE_LSTM = s.MODEL_WEIGHT_FILE_LSTM

    GEOJSON_FILE_TURE = s.GEOJSON_FILE_TRUE
    GEOJSON_FILE_PREDICTED_LSTM = s.GEOJSON_FILE_PREDICTED_LSTM
    GEOJSON_FILE_PREDICTED_VELOCITY = s.GEOJSON_FILE_PREDICTED_VELOCITY

    (X_train, y_train), (X_test, y_test), (X_scaler, y_scaler) = load_dataset.load_dataset(TRANSFER_CSV_FILTERED, EXPERIMENT_PARAMETERS)
    y_predicted_lstm = predict.prediction_lstm(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, MODEL_FILE_LSTM, MODEL_WEIGHT_FILE_LSTM, FIGURE_DIR)
    y_predicted_velocity = predict.prediction_velocity(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS)

    X_test = load_dataset.inverse_scale_transform_sample(X_test, X_scaler, X=True)
    y_test = load_dataset.inverse_scale_transform_sample(y_test, y_scaler)
    y_predicted_lstm = load_dataset.inverse_scale_transform_sample(y_predicted_lstm, y_scaler)
    y_predicted_velocity = load_dataset.inverse_scale_transform_sample(y_predicted_velocity, y_scaler)

    logger.info("Accuracy of LSTM")
    predict.calculate_rmse_on_array(y_predicted_lstm, y_test)
    logger.info("Accuracy of velocity")
    predict.calculate_rmse_on_array(y_predicted_velocity, y_test)

    predict.save_numpy_array(X_test, X_FILE)
    predict.save_numpy_array(y_test, Y_FILE)
    predict.save_numpy_array(y_predicted_lstm, Y_FILE_PREDICTED_LSTM)
    predict.save_numpy_array(y_predicted_velocity, Y_FILE_PREDICTED_VELOCITY)


    X_test = numpy.load(X_FILE)
    y_test = numpy.load(Y_FILE)
    y_predicted_lstm = numpy.load(Y_FILE_PREDICTED_LSTM)
    y_predicted_velocity = numpy.load(Y_FILE_PREDICTED_VELOCITY)

    predict.create_geojson_line(X_test, y_test, GEOJSON_FILE_TURE, EXPERIMENT_PARAMETERS)
    predict.create_geojson_prediction_line(X_test, y_predicted_lstm, GEOJSON_FILE_PREDICTED_LSTM, EXPERIMENT_PARAMETERS)
    predict.create_geojson_prediction_line(X_test, y_predicted_velocity, GEOJSON_FILE_PREDICTED_VELOCITY, EXPERIMENT_PARAMETERS)
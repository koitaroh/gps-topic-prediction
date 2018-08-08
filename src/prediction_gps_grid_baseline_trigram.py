import requests
import json
import csv
import os
import time
import random
import math
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
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed, LSTM, SimpleRNN, GRU, Input, Flatten, Bidirectional
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

import settings as s
import load_dataset
import preprocessing_gps

np.random.seed(7)
random.seed(7)



def define_dictionary():
    return None



if __name__ == '__main__':
    slack_client = s.sc
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR
    X_GRID_FILE = s.X_GRID_FILE
    Y_GRID_FILE = s.Y_GRID_FILE

    X_FILE = s.X_FILE
    Y_FILE = s.Y_FILE
    Y_FILE_PREDICTED_LSTM = s.Y_FILE_PREDICTED_LSTM
    Y_FILE_PREDICTED_VELOCITY = s.Y_FILE_PREDICTED_VELOCITY

    MODEL_FILE_LSTM_GRID = s.MODEL_FILE_LSTM_GRID
    MODEL_WEIGHT_FILE_LSTM_GRID = s.MODEL_WEIGHT_FILE_LSTM_GRID

    GEOJSON_FILE_OBSERVATION_GRID = s.GEOJSON_FILE_OBSERVATION_GRID
    GEOJSON_FILE_TRUE_GRID = s.GEOJSON_FILE_TRUE_GRID
    GEOJSON_FILE_PREDICTED_LSTM_GRID = s.GEOJSON_FILE_PREDICTED_LSTM_GRID

    define_dictionary()

    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__+" is finished.")
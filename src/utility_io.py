import requests
import json
import csv
import multiprocessing as mp
from itertools import repeat
import os
import sys
import time
import random
import sqlalchemy
import pandas
import tables
import pickle
import numpy
import numpy as np
from datetime import datetime, timedelta
import geopy
from geopy.distance import vincenty
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import smart_open
import random
import multiprocessing
import pymysql
import configparser
from tqdm import tqdm
import geojson
from geojson import LineString, FeatureCollection, Feature
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError

# Logging ver. 2017-12-14
from logging import handlers
import logging
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


def load_predicted_trajectory_csv_to_dataframe(CSV_FILE):
    logger.info("Loading CSV %s to dataframe" % CSV_FILE)
    headers = ['uid', 'timestamp', 'x', 'y']
    dtype = {'uid': 'int', 'timestamp': 'str', 'x': 'float64', 'y': 'float64'}
    parse_dates = ['timestamp']


    df_csv = pandas.read_csv(filepath_or_buffer=CSV_FILE, header=None, names=headers,
                             dtype=dtype, parse_dates=parse_dates,
                             usecols=[0, 1, 2, 3], error_bad_lines=False, warn_bad_lines=True)
    return df_csv


def load_csv_files_to_dataframe(DATA_DIR, EXPERIMENT_PARAMETERS):
    timestart = datetime.strptime(EXPERIMENT_PARAMETERS['TIMESTART'], '%Y-%m-%d %H:%M:%S')
    timeend = datetime.strptime(EXPERIMENT_PARAMETERS['TIMEEND'], '%Y-%m-%d %H:%M:%S')
    days = (timeend - timestart).days
    prediction_days = days + 1
    df_raw_all = pd.DataFrame(np.empty(0, dtype=[('uid', 'str'), ('time_start', 'str'), ('time_end', 'str'), ('x', 'float64'), ('y', 'float64'), ('mode', 'str')]))
    num_user_all = 0
    for root, dirs, files in os.walk(DATA_DIR):
        files.sort()
        for fn in files:
            if fn[0] != '.':
                csv_file = root + "/" + fn
                df, num_user = load_csv_to_dataframe(csv_file, EXPERIMENT_PARAMETERS)
                df_raw_all = df_raw_all.append(df)
                num_user_all += num_user
                logger.info("Current number of user: %s" % num_user_all)
                if EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE'] * 2 < num_user_all:
                    break
    # df_raw_all = df_raw_all.sort_values(by=['uid', 'time_start'])
    return df_raw_all


def load_csv_to_dataframe(CSV_FILE, EXPERIMENT_PARAMETERS):
    logger.info("Loading CSV %s to dataframe" % CSV_FILE)
    headers = ['uid', 'time_start', 'time_end', 'x', "y", "mode"]
    dtype = {'uid': 'str', 'time_start': 'str', "time_end": "str", "x": "float64", 'y': 'float64', 'mode': 'str'}
    parse_dates = ['time_start', 'time_end']
    df_csv = pd.read_csv(filepath_or_buffer=CSV_FILE, header=None, names=headers,
                             dtype=dtype, parse_dates=parse_dates, error_bad_lines=False, warn_bad_lines=True)
    # df_csv = df_csv.sort_values(by=['uid', 'timestamp'])
    # logger.info("Filtering dataframe with experiment parameters")
    # grouped = df_csv.groupby('uid')
    # logger.info("Number of uid: %s" % len(grouped))

    # logger.info("Filtering with time setting")
    df_filtered = df_csv[(df_csv['time_end'] >= EXPERIMENT_PARAMETERS["TIMESTART"]) & (df_csv['time_start'] <= EXPERIMENT_PARAMETERS["TIMEEND"])]
    grouped = df_filtered.groupby('uid')
    num_user = len(grouped)
    # logger.info("Number of uid after temporal filter: %s" % num_user)
    return df_filtered, num_user


def save_dataframe_to_csv(df: pd.DataFrame, csv_file, regularized=False):
    logger.info("Saving dataframe to CSV %s" % csv_file)
    if (regularized):
        df.to_csv(csv_file, columns=['uid', 'timestamp', 't_index', 'y', 'x', 'mode', 's_index'], index=False)
    else:
        df.to_csv(csv_file, columns=['uid', 'timestamp', 'y', 'x'], index=False)
    return None


if __name__ == '__main__':
    TEST_DIR = "/Users/koitaroh/Downloads/training/"
    TEST_DROPBOX_PATH = "/training/"

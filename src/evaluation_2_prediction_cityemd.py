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
import pandas as pd
import pickle
import numpy as np
import h5py
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
import math
from decimal import Decimal
from collections import Counter
from tqdm import tqdm
from pyemd import emd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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
import load_dataset
import jpgrid
import utility_io
import utility_spatiotemporal_index
import prediction_6_gps_grid_mode_topic

np.random.seed(1)
random.seed(1)


def evaluate_prediction_cityemd_example():
    first_histogram = np.array([0.0, 1.0])
    second_histogram = np.array([5.0, 3.0])
    distance_matrix = np.array([[0.0, 0.5],[0.5, 0.0]])
    print(emd(first_histogram, second_histogram, distance_matrix))


def evaluate_prediction_cityemd(MESHCODE_FILE_2ND, CITYEMD_DISTANCE_MATRIX_FILE, Y_PREDICTED_FILE, Y_TRUE_FILE):
    cityemd_prediction_dict = {}
    df_mesh = pd.read_csv(MESHCODE_FILE_2ND, delimiter=',', header=None, dtype='str')
    mesh_list = df_mesh[0].values.tolist()
    y_predicted = utility_io.load_predicted_trajectory_csv_to_dataframe(Y_PREDICTED_FILE)
    y_true = utility_io.load_predicted_trajectory_csv_to_dataframe(Y_TRUE_FILE)

    first_histogram = np.zeros(len(mesh_list))
    second_histogram = np.zeros(len(mesh_list))
    distance_matrix = np.load(CITYEMD_DISTANCE_MATRIX_FILE)

    y_predicted_grouped = y_predicted.groupby('timestamp')
    y_true_grouped = y_true.groupby('timestamp')

    for name, y_predicted_group in y_predicted_grouped:

        y_predicted_xy_list = y_predicted_group[['x', 'y']].values.tolist()

        y_true_group = y_true_grouped.get_group(name)
        y_true_xy_list = y_true_group[['x', 'y']].values.tolist()


        for x_predicted, y_predicted in y_predicted_xy_list:
            meshcode_predicted = jpgrid.encodeLv2(y_predicted, x_predicted)
            first_histogram[mesh_list.index(meshcode_predicted)] += 1

        for x, y in y_true_xy_list:
            meshcode = jpgrid.encodeLv2(y, x)
            second_histogram[mesh_list.index(meshcode)] += 1

        # print(first_histogram)
        # print(second_histogram)

        city_emd = emd(first_histogram, second_histogram, distance_matrix)

        # print(city_emd)
        # if city_emd > 0:
        #     print(math.log(city_emd))

        # print(f"CityEMD value at time {name}: {city_emd}")
        if city_emd > 0:
            logger.info(f"CityEMD Log value at time {name}: {math.log(city_emd)}")
            cityemd_prediction_dict[name] = math.log(city_emd)
    return cityemd_prediction_dict


def evaluate_dummy_cityemd(MESHCODE_FILE_2ND, CITYEMD_DISTANCE_MATRIX_FILE, X_OBSERVATION_FILE, Y_TRUE_FILE):
    cityemd_dummy_dict = {}
    df_mesh = pd.read_csv(MESHCODE_FILE_2ND, delimiter=',', header=None, dtype='str')
    mesh_list = df_mesh[0].values.tolist()
    y_true = utility_io.load_predicted_trajectory_csv_to_dataframe(Y_TRUE_FILE)
    x_observation = utility_io.load_predicted_trajectory_csv_to_dataframe(X_OBSERVATION_FILE)

    first_histogram = np.zeros(len(mesh_list))
    second_histogram = np.zeros(len(mesh_list))
    distance_matrix = np.load(CITYEMD_DISTANCE_MATRIX_FILE)

    y_true_grouped = y_true.groupby('timestamp')
    x_observation_grouped = x_observation.groupby('timestamp')

    for name, x_obs in x_observation_grouped:
        x_observation_group = x_observation_grouped.get_group(name)
        x_observation_last = x_observation_group[['x', 'y']].values.tolist()
    for x, y in x_observation_last:
        meshcode = jpgrid.encodeLv2(y, x)
        second_histogram[mesh_list.index(meshcode)] += 1
    for name, y_true_group in y_true_grouped:
        y_true_group = y_true_grouped.get_group(name)
        y_true_xy_list = y_true_group[['x', 'y']].values.tolist()

        for x, y in y_true_xy_list:
            meshcode = jpgrid.encodeLv2(y, x)
            first_histogram[mesh_list.index(meshcode)] += 1

        city_emd = emd(first_histogram, second_histogram, distance_matrix)

        # print(city_emd)
        # if city_emd > 0:
        #     print(math.log(city_emd))
        # print(f"CityEMD value at time {name}: {city_emd}")
        if city_emd > 0:
            # print(f"CityEMD Log value at time {name}: {math.log(city_emd)}")
            logger.info(f"CityEMD Log value at time {name}: {math.log(city_emd)}")
            cityemd_dummy_dict[name] = math.log(city_emd)
    return cityemd_dummy_dict


def draw_cityemd_lineplot(cityemd_prediction_dict, cityemd_dummy_dict, FIGURE_DIR):
    fig, ax = plt.subplots(figsize=(10, 7))
    dates = list(cityemd_prediction_dict.keys())
    cityemd_prediction = list(cityemd_prediction_dict.values())
    plt.plot_date(dates, cityemd_prediction, '-', label='prediction')
    dates = list(cityemd_dummy_dict.keys())
    cityemd_dummy = list(cityemd_dummy_dict.values())
    plt.plot_date(dates, cityemd_dummy, '-', label='dummy')
    plt.xlabel('Timestamp')
    plt.ylabel('CityEMD (Log)')
    plt.legend(loc='upper left')
    # plt.margins(0,0)
    plt.title("CityEMD w.r.t. time")
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    plt.savefig(FIGURE_DIR / "city_emd.pdf", dpi=300)
    plt.savefig(FIGURE_DIR / "city_emd.png", dpi=300)
    return None


if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR
    CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID
    CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID
    CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID = s.CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID

    MESHCODE_FILE_2ND = s.MESHCODE_FILE_2ND
    CITYEMD_DISTANCE_MATRIX_FILE = s.CITYEMD_DISTANCE_MATRIX_FILE

    logger.info('Calculating LSTM CityEMD (Raw -> Log)')
    cityemd_prediction_dict = evaluate_prediction_cityemd(MESHCODE_FILE_2ND, CITYEMD_DISTANCE_MATRIX_FILE, CSV_TRAJECTORY_FILE_EVALUATION_PREDICTED_GRID, CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID)
    logger.info('Calculating Dummy CityEMD (Raw -> Log)')
    cityemd_dummy_dict = evaluate_dummy_cityemd(MESHCODE_FILE_2ND, CITYEMD_DISTANCE_MATRIX_FILE, CSV_TRAJECTORY_FILE_EVALUATION_OBSERVATION_GRID, CSV_TRAJECTORY_FILE_EVALUATION_TRUE_GRID)

    draw_cityemd_lineplot(cityemd_prediction_dict, cityemd_dummy_dict, FIGURE_DIR)

    elapsed = str((datetime.now() - start_time))
    logger.info(f"Finished in: {elapsed}")
    atexit.register(s.exit_handler, __file__, elapsed)

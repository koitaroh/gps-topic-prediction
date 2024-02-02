import os
import time
import random
import tables
import pickle
import requests
import json
import csv
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import repeat
from datetime import datetime, timedelta
import geopy
from geopy.distance import geodesic
import smart_open
import random
import configparser

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
import jpgrid


def define_temporal_index(EXPERIMENT_PARAMETERS):
    logger.info("Defining temporal indices")
    temporal_indices = []
    temporal_index = 0
    timestart = datetime.strptime(EXPERIMENT_PARAMETERS["TRAINING_OBSERVATION_START"], '%Y-%m-%d %H:%M:%S')
    timeend = datetime.strptime(EXPERIMENT_PARAMETERS["TRAINING_PREDICTION_END"], '%Y-%m-%d %H:%M:%S')
    unit_temporal = timedelta(minutes=EXPERIMENT_PARAMETERS["UNIT_TEMPORAL"])
    time_cursor = timestart
    while (time_cursor < timeend):
        # time_cursor_str = datetime.strftime(time_cursor, '%Y-%m-%d %H:%M:%S')
        iso_year, iso_week_number, iso_weekday = time_cursor.isocalendar()
        temporal_indices.append([temporal_index, iso_year, iso_week_number, iso_weekday, time_cursor])
        # temporal_indices[temporal_index] = time_cursor
        temporal_index += 1
        time_cursor = time_cursor + unit_temporal
        # logger.debug(time_cursor_str)
    return temporal_indices


# define_temporal_slices
def define_temporal_index_evaluation(EXPERIMENT_PARAMETERS):
    logger.info("Defining temporal indices for evaluation")
    temporal_indices = []
    temporal_index = 0
    timestart = datetime.strptime(EXPERIMENT_PARAMETERS["EVALUATION_OBSERVATION_START"], '%Y-%m-%d %H:%M:%S')
    timeend = datetime.strptime(EXPERIMENT_PARAMETERS["EVALUATION_PREDICTION_END"], '%Y-%m-%d %H:%M:%S')
    unit_temporal = timedelta(minutes=EXPERIMENT_PARAMETERS["UNIT_TEMPORAL"])
    time_cursor = timestart
    while (time_cursor < timeend):
        # time_cursor_str = datetime.strftime(time_cursor, '%Y-%m-%d %H:%M:%S')
        iso_year, iso_week_number, iso_weekday = time_cursor.isocalendar()
        temporal_indices.append([temporal_index, iso_year, iso_week_number, iso_weekday, time_cursor])
        # temporal_indices[temporal_index] = time_cursor
        temporal_index += 1
        time_cursor = time_cursor + unit_temporal
        # logger.debug(time_cursor_str)
    return temporal_indices


def define_spatial_index(EXPERIMENT_PARAMETERS):
    x1y1 = (EXPERIMENT_PARAMETERS["AOI"][1], EXPERIMENT_PARAMETERS["AOI"][0])
    x2y1 = (EXPERIMENT_PARAMETERS["AOI"][1], EXPERIMENT_PARAMETERS["AOI"][2])
    x1y2 = (EXPERIMENT_PARAMETERS["AOI"][3], EXPERIMENT_PARAMETERS["AOI"][0])
    x2y2 = (EXPERIMENT_PARAMETERS["AOI"][3], EXPERIMENT_PARAMETERS["AOI"][2])
    x_distance = geodesic(x1y1, x2y1).meters
    y_distance = geodesic(x1y1, x1y2).meters
    logger.debug("X distance: %s meters, Y distance: %s meters", x_distance, y_distance)
    x_unit_degree = round((((EXPERIMENT_PARAMETERS["AOI"][2] - EXPERIMENT_PARAMETERS["AOI"][0]) * EXPERIMENT_PARAMETERS["UNIT_SPATIAL_METER"]) / x_distance), 4)
    y_unit_degree = round((((EXPERIMENT_PARAMETERS["AOI"][3] - EXPERIMENT_PARAMETERS["AOI"][1]) * EXPERIMENT_PARAMETERS["UNIT_SPATIAL_METER"]) / y_distance), 4)
    logger.debug("X unit in degree: %s degrees, Y unit in degree: %s degrees", x_unit_degree, y_unit_degree)
    x_size = int((EXPERIMENT_PARAMETERS["AOI"][2] - EXPERIMENT_PARAMETERS["AOI"][0]) // x_unit_degree) + 1
    y_size = int((EXPERIMENT_PARAMETERS["AOI"][3] - EXPERIMENT_PARAMETERS["AOI"][1]) // y_unit_degree) + 1
    logger.info("X size: %s", x_size)
    logger.info("Y size: %s", y_size)
    logger.info("Size of spatial index: %s", x_size * y_size)
    spatial_index = [x_unit_degree, y_unit_degree, x_size, y_size]
    return spatial_index


def convert_raw_coordinates_to_spatial_index(EXPERIMENT_PARAMETERS, spatial_index, x, y):
    x_index = int((x - EXPERIMENT_PARAMETERS["AOI"][0]) // spatial_index[0])
    y_index = int((y - EXPERIMENT_PARAMETERS["AOI"][1]) // spatial_index[1])
    print(x_index)
    print(y_index)
    spatial_index_number = x_index + (spatial_index[3] * y_index)
    return spatial_index_number


def convert_spatial_index_to_raw_coordinates(EXPERIMENT_PARAMETERS, spatial_index, spatial_index_number):
    spatial_index_number = int(spatial_index_number)
    y_index = spatial_index_number // spatial_index[3]
    x_index = spatial_index_number - (y_index * spatial_index[3])
    # print(x_index)
    # print(y_index)
    x = EXPERIMENT_PARAMETERS["AOI"][0] + (spatial_index[0] * x_index) + (spatial_index[0] / 2)
    y = EXPERIMENT_PARAMETERS["AOI"][1] + (spatial_index[1] * y_index) + (spatial_index[1] / 2)
    return x, y


def convert_spatial_index_to_latitude(meshcode):
    y, x = jpgrid.decode(meshcode)
    return y


def convert_spatial_index_to_longitude(meshcode):
    y, x = jpgrid.decode(meshcode)
    return x

def calculate_rmse_in_meters(input_array):
    origin = (input_array[0], input_array[1])
    destination = (input_array[2], input_array[3])
    distance = geodesic(origin, destination).meters
    return distance

if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    temporal_index = define_temporal_index(EXPERIMENT_PARAMETERS)
    logger.info(temporal_index)
    logger.info(len(temporal_index))
    spatial_index = define_spatial_index(EXPERIMENT_PARAMETERS)
    logger.info(spatial_index)
    logger.info(len(spatial_index))
    logger.info(EXPERIMENT_PARAMETERS["AOI"])
    X_TEST = 138.71
    Y_TEST = 34.90

    print(jpgrid.encodeBase(35.65486522, 156.42456722)) # 53563383
    print(jpgrid.decode('53563383')) # (35.65416666666667, 156.41875)

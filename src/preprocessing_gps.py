import requests
import json
import csv
import os
import time
import random
import multiprocessing as mp
from itertools import repeat
import sqlalchemy
import pandas
import pickle
import numpy
import numpy as np
import datetime
from datetime import datetime, timedelta
import geopy
from geopy.distance import vincenty
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

np.random.seed(7)
random.seed(7)


def load_csv_files_to_dataframe(DATA_DIR, EXPERIMENT_PARAMETERS):
    timestart = datetime.strptime(EXPERIMENT_PARAMETERS['TIMESTART'], '%Y-%m-%d %H:%M:%S')
    timeend = datetime.strptime(EXPERIMENT_PARAMETERS['TIMEEND'], '%Y-%m-%d %H:%M:%S')
    days = (timeend - timestart).days
    prediction_days = days + 2
    logger.info("Loading %s days" % prediction_days)
    df_raw_all = pandas.DataFrame(numpy.empty(0, dtype=[('uid', 'str'), ('timestamp', 'str'), ('y', 'float64'), ('x', 'float64')]))

    for i in range(prediction_days):
        new_date = timestart + timedelta(days=i)
        timestart_text = datetime.strftime(new_date, '%Y%m%d')
        csv_file = DATA_DIR + timestart_text + ".csv"
        df = load_csv_to_dataframe(csv_file)
        df_raw_all = df_raw_all.append(df)
    df_raw_all = df_raw_all.sort_values(by=['uid', 'timestamp'])
    return df_raw_all


def load_csv_to_dataframe(CSV_FILE, regularized=False, sns=False):
    logger.info("Loading CSV %s to dataframe" % CSV_FILE)
    if (regularized):
        headers = ['uid', 't_index', 'iso_year', 'iso_week_number', 'iso_weekday', 'timestamp', 'y', 'x']
        # dtype = {'uid': 'str', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}
        dtype = {'uid': 'str', 't_index': 'int64', 'iso_year': 'int64', 'iso_week_number': 'int64', 'iso_weekday': 'int64', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}

        parse_dates = ['timestamp']
        df_csv = pandas.read_csv(filepath_or_buffer=CSV_FILE, header=None, names=headers,
                                 dtype=dtype, parse_dates=parse_dates, skiprows=1,
                                 usecols=[0, 1, 2, 3, 4, 5, 6, 7], error_bad_lines=False, warn_bad_lines=True)
    elif (sns):
        headers = ['uid', 'timestamp', 'y', 'x']
        dtype = {'uid': 'str', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}
        parse_dates = ['timestamp']
        df_csv = pandas.read_csv(filepath_or_buffer=CSV_FILE, header=None, names=headers,
                                 dtype=dtype, parse_dates=parse_dates,
                                 usecols=[0, 2, 3, 4], error_bad_lines=False, warn_bad_lines=True)
    else:
        headers = ['uid', 'timestamp', 'y', 'x']
        dtype = {'uid': 'str', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}
        parse_dates = ['timestamp']
        df_csv = pandas.read_csv(filepath_or_buffer=CSV_FILE, header=None, names=headers,
                                 dtype=dtype, parse_dates=parse_dates,
                                 usecols=[0,1,2,3], error_bad_lines=False, warn_bad_lines=True)
    # df_csv = df_csv.sort_values(by=['uid', 'timestamp'])
    return df_csv


# save_dataframe_to_csv
def save_dataframe_to_csv(df: pandas.DataFrame, csv_file, regularized=False):
    logger.info("Saving dataframe to CSV %s" % csv_file)
    if (regularized):
        df.to_csv(csv_file, columns=['uid', 't_index', 'iso_year', 'iso_week_number', 'iso_weekday', 'timestamp', 'y', 'x', 's_index'], index=False)
    else:
        df.to_csv(csv_file, columns=['uid', 'timestamp', 'y', 'x'], index=False)
    return None


def filter_users_with_experiment_setting(DATAFRAME, EXPERIMENT_PARAMETERS):
    logger.info("Filtering dataframe with experiment parameters")
    df_users_filtered = DATAFRAME
    # print(df_user_filtered)
    traj_set = dict({})
    grouped = df_users_filtered.groupby('uid')
    logger.info("Number of uid: %s" % len(grouped))
    logger.info("Filtering with AOI")
    df_users_filtered = grouped.filter(lambda x: x['x'].min()>= float(EXPERIMENT_PARAMETERS["AOI"][0]))
    df_users_filtered = df_users_filtered.groupby('uid').filter(lambda x: x['x'].max()<= float(EXPERIMENT_PARAMETERS["AOI"][2]))
    df_users_filtered = df_users_filtered.groupby('uid').filter(lambda x: x['y'].min()>= float(EXPERIMENT_PARAMETERS["AOI"][1]))
    df_users_filtered = df_users_filtered.groupby('uid').filter(lambda x: x['y'].max()<= float(EXPERIMENT_PARAMETERS["AOI"][3]))
    logger.info("Number of uid after spatial filter: %s" % len(df_users_filtered.groupby('uid')))

    # print(df_user_filtered)
    logger.info("Filtering with time setting")
    df_users_filtered = df_users_filtered[(df_users_filtered['timestamp'] >= EXPERIMENT_PARAMETERS["TIMESTART"]) & (df_users_filtered['timestamp'] <= EXPERIMENT_PARAMETERS["TIMEEND"])]
    # print(df_user_filtered)
    logger.info("Number of uid after temporal filter: %s" % len(df_users_filtered.groupby('uid')))
    logger.info("Filtering with number of sample user")
    grouped = df_users_filtered.groupby('uid')
    num_user = len(grouped)
    if EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE'] <= num_user:
        num_user = EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE']
    sampled_df_i = random.sample(list(grouped.indices), num_user)
    df_list = map(lambda df_i: grouped.get_group(df_i), sampled_df_i)
    df_users_filtered = pandas.concat(df_list, axis=0, join='outer')
    logger.info("Number of uid after sample user filter: %s" % len(df_users_filtered.groupby('uid')))

    return df_users_filtered


# define_temporal_slices
def define_temporal_index(EXPERIMENT_PARAMETERS):
    logger.info("Defining temporal indices")
    temporal_indices = []
    # temporal_indices = dict()
    temporal_index = 0
    timestart = datetime.strptime(EXPERIMENT_PARAMETERS["TIMESTART"], '%Y-%m-%d %H:%M:%S')
    timeend = datetime.strptime(EXPERIMENT_PARAMETERS["TIMEEND"], '%Y-%m-%d %H:%M:%S')
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
    x_distance = geopy.distance.vincenty(x1y1, x2y1).meters
    y_distance = geopy.distance.vincenty(x1y1, x1y2).meters
    logger.debug("X distance: %s meters, Y distance: %s meters", x_distance, y_distance)
    x_unit_degree = round((((EXPERIMENT_PARAMETERS["AOI"][2] - EXPERIMENT_PARAMETERS["AOI"][0]) * EXPERIMENT_PARAMETERS["UNIT_SPATIAL_METER"]) / x_distance), 4)
    y_unit_degree = round((((EXPERIMENT_PARAMETERS["AOI"][3] - EXPERIMENT_PARAMETERS["AOI"][1]) * EXPERIMENT_PARAMETERS["UNIT_SPATIAL_METER"]) / y_distance), 4)
    logger.debug("X unit in degree: %s degrees, Y unit in degree: %s degrees", x_unit_degree, y_unit_degree)
    x_size = int((EXPERIMENT_PARAMETERS["AOI"][2] - EXPERIMENT_PARAMETERS["AOI"][0]) // x_unit_degree) + 1
    y_size = int((EXPERIMENT_PARAMETERS["AOI"][3] - EXPERIMENT_PARAMETERS["AOI"][1]) // y_unit_degree) + 1
    logger.info("X size: %s", x_size)
    logger.info("Y size: %s", y_size)
    logger.info("Size of spatial index: %s", x_size * y_size)
    # t_start = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    # t_end = datetime.datetime.strptime(timeend, '%Y-%m-%d %H:%M:%S')
    # t_size = round((t_end - t_start) / datetime.timedelta(minutes=unit_temporal))
    # logger.info("T size: %s", t_size)
    # logger.info("Spatiotemporal units: %s", [t_size, x_size, y_size, num_topic])
    spatial_index = [x_unit_degree, y_unit_degree, x_size, y_size]
    return spatial_index


def convert_raw_coordinates_to_spatial_index(EXPERIMENT_PARAMETERS, spatial_index, x, y):
    x_index = int((x - EXPERIMENT_PARAMETERS["AOI"][0]) // spatial_index[0])
    y_index = int((y - EXPERIMENT_PARAMETERS["AOI"][1]) // spatial_index[1])
    # print(x_index)
    # print(y_index)
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


# apply slice to users
def apply_slice_to_users(df_users_filtered: pandas.DataFrame, temporal_indices, EXPERIMENT_PARAMETERS):
    logger.info("Applying slice to users")
    df_users_regularized = pandas.DataFrame(numpy.empty(0, dtype=[('uid', 'int32'), ('t_index', 'int32'), ('iso_year', 'int32'), ('iso_week_number', 'int32'),('iso_weekday', 'int32'), ('timestamp', 'str'), ('y', 'float64'), ('x', 'float64'), ('s_index', 'int32')]))

    # print(df_users_regularized)
    headers = ['uid', 't_index', 'iso_year', 'iso_week_number', 'iso_weekday', 'timestamp', 'y', 'x', 's_index']
    # df_users_regularized = pandas.DataFrame(columns=headers)

    grouped = df_users_filtered.groupby('uid')

    for name, group in grouped:
        traj = []
        # print(name)
        # print(group)
        # for each user, convert trajectory to constant normalized trajectory.
        for index, row in group.iterrows():
            # timestamp = time.mktime(time.strftime(row.timestamp, '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
            timestamp = pandas.to_datetime(row.timestamp)
            traj.append([name, timestamp, row.y, row.x])
        # print(traj)
        traj_slice = slice_trajectory(traj, temporal_indices, EXPERIMENT_PARAMETERS)
        # print(traj_slice)
        df_user_regularized = pandas.DataFrame.from_records(traj_slice, columns=headers)
        # print(df_user_regularized)
        df_users_regularized = df_users_regularized.append(df_user_regularized)
    # print(df_users_regularized)
    return df_users_regularized


def apply_slice_to_users_multiprocessing(df_users_filtered: pandas.DataFrame, temporal_indices, spatial_index, EXPERIMENT_PARAMETERS, pool):
    logger.info("Applying slice to users")
    df_users_regularized = pandas.DataFrame(numpy.empty(0, dtype=[('uid', 'int32'), ('t_index', 'int32'), ('iso_year', 'int32'), ('iso_week_number', 'int32'),('iso_weekday', 'int32'), ('timestamp', 'str'), ('y', 'float64'), ('x', 'float64'), ('s_index', 'int32')]))
    headers = ['uid', 't_index', 'iso_year', 'iso_week_number', 'iso_weekday', 'timestamp', 'y', 'x', 's_index']
    # df_users_regularized = pandas.DataFrame(columns=headers)
    grouped = df_users_filtered.groupby('uid')
    df_list = pool.starmap(slice_user_trajectory, zip(grouped, repeat(temporal_indices), repeat(spatial_index), repeat(EXPERIMENT_PARAMETERS)))
    df_users_regularized = pandas.concat(df_list)
    return df_users_regularized


def slice_user_trajectory(grouped, temporal_index, spatial_index, EXPERIMENT_PARAMETERS):
    name, group = grouped
    traj = []
    df_user_regularized = pandas.DataFrame(numpy.empty(0, dtype=[('uid', 'int32'), ('timestamp', 'str'), ('t_index', 'int32'), ('y', 'float64'), ('x', 'float64'), ('mode', 'str'), ('s_index', 'int32')]))
    for index, row in group.iterrows():
        timestamp = pandas.to_datetime(row.timestamp)
        traj.append([name, timestamp, row.y, row.x])
    traj_slice = slice_trajectory(traj, temporal_index, spatial_index, EXPERIMENT_PARAMETERS)
    headers = ['uid', 't_index', 'iso_year', 'iso_week_number', 'iso_weekday', 'timestamp', 'y', 'x', 's_index']
    df_user_regularized = pandas.DataFrame.from_records(traj_slice, columns=headers)
    # df_user_regularized = df_users_regularized.append(df_user_regularized)
    # print(df_user_regularized)
    return df_user_regularized


def slice_trajectory(traj_original: list, temporal_indices: list, spatial_index, EXPERIMENT_PARAMETERS):
    traj_slice = []
    uid = traj_original[0][0]

    for t_index, iso_year, iso_week_number, iso_weekday, t in temporal_indices:
        y, x, spatial_index_number = estimate_location_at_time(traj_original, t, spatial_index, EXPERIMENT_PARAMETERS)
        time_step = datetime.strftime(t, '%Y-%m-%d %H:%M:%S')  # Make it String
        traj_slice.append([uid, t_index, iso_year, iso_week_number, iso_weekday, time_step, y, x, spatial_index_number])


    # for time_step in temporal_indices:
        # y, x = estimate_location_at_time(traj_original, time_step)
        # time_step = datetime.strftime(time_step, '%Y-%m-%d %H:%M:%S') # Make it String
        # traj_slice.append([time_step, y, x])


    return traj_slice


def estimate_location_at_time(traj: list, t: datetime.timestamp, spatial_index, EXPERIMENT_PARAMETERS):
    t0_uid, t0_t, t0_lat, t0_lon = traj[0]
    t1_uid, t1_t, t1_lat, t1_lon = traj[0]
    lat = t0_lat
    lon = t0_lon

    for idx, val in enumerate(traj):
        if idx == 0:
            t0_uid, t0_t, t0_lat, t0_lon = val
            t1_uid, t1_t, t1_lat, t1_lon = val
        else:
            t0_uid, t0_t, t0_lat, t0_lon = traj[idx-1]
            t1_uid, t1_t, t1_lat, t1_lon = val
            # print(traj[idx - 1])
            # print(val)
            if t1_t > t: # If the timespan crosses the target
                # print(traj[idx - 1])
                # print(val)
                dt = t1_t - t0_t
                # print(dt)
                dt = timedelta.total_seconds(dt)
                # print(dt)
                # print(type(dt))
                dlat = t1_lat - t0_lat
                dlon = t1_lon - t0_lon
                # print(dlat)
                lat = t0_lat + timedelta.total_seconds(t - t0_t) * dlat / (dt + 0.0000001)
                lon = t0_lon + timedelta.total_seconds(t - t0_t) * dlon / (dt + 0.0000001)
                break
    spatial_index_number = convert_raw_coordinates_to_spatial_index(EXPERIMENT_PARAMETERS, spatial_index, lon, lat)
    return lat, lon, spatial_index_number


def estimate_location_at_time_old(traj: list, t):
    print(t)
    # traj_est = {}
    # t = time.mktime(time.strptime(t_str, '%Y-%m-%d %H:%M:%S'))
    # t = time.mktime(time.strptime(time_moment, '%Y-%m-%d %H:%M:%S'))
    pre_rec = traj[0]
    cur_rec = traj[0]
    for rec in traj:
        cur_uid, cur_t, cur_lat, cur_lon  = rec
        pre_uid, pre_t, pre_lat, pre_lon  = pre_rec
        # print(cur_t)

        if cur_t < t:
            pre_rec = cur_rec
            cur_rec = rec
        else:
            dt = cur_t - pre_t
            # print(dt)
            dt = timedelta.total_seconds(dt)
            # print(dt)

            # print(type(dt))
            dlat = cur_lat - pre_lat
            dlon = cur_lon - pre_lon
            # print(cur_lat)
            # print(pre_lat)
            # print(dlat)
            lat = pre_lat + timedelta.total_seconds(t - pre_t) * dlat / (dt + 0.0000001)
            lon = pre_lon + timedelta.total_seconds(t - pre_t) * dlon / (dt + 0.0000001)
            return lat, lon
        last_uid, last_t, lat, lon = cur_rec
    return lat, lon


def convert_dataframe_to_coordinate_sequences(df: pandas.DataFrame, EXPERIMENT_PARAMETERS):
    # df = df.drop('timestamp', 1)
    # df = df.drop('t_index', 1)
    # df = df.drop('iso_year', 1)
    # df = df.drop('iso_week_number', 1)
    # df = df.drop('iso_weekday', 1)

    grouped = df.groupby('uid')
    num_user = len(grouped)
    sample_size = EXPERIMENT_PARAMETERS['SAMPLE_SIZE']
    recall_length = EXPERIMENT_PARAMETERS['RECALL_LENGTH']
    predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    sample_size_per_user = sample_size // num_user
    logger.info("Sample size per user: %s" % sample_size_per_user)
    X_all = []
    y_all = []
    for name, group in grouped:
        # logger.debug(name)
        # print(group)
        # group.reset_index()
        # print(group.iloc[0:1])
        # print(len(group))
        user_random_index = numpy.random.randint(low=0, high=(len(group)-(recall_length + predict_length)), size=sample_size_per_user)
        print(user_random_index)
        for i in numpy.nditer(user_random_index):
            i = int(i)
            x_user_all = group.iloc[i:i+recall_length, -3:-1].values.tolist()
            y_user_all = group.iloc[i+recall_length:i+recall_length+predict_length, -3:-1].values.tolist()
            X_all.append(x_user_all)
            if predict_length == 1:
                y_all.append(y_user_all[0])
            else:
                y_all.append(y_user_all)
    X_all = numpy.array(X_all)
    y_all = numpy.array(y_all)
    print('X_all shape:', X_all.shape)
    # print(X_all)
    print('y_all shape:', y_all.shape)
    # print(y_all)
    return X_all, y_all


def convert_dataframe_to_grid_sequences(df: pandas.DataFrame, EXPERIMENT_PARAMETERS):
    grouped = df.groupby('uid')
    num_user = len(grouped)
    sample_size = EXPERIMENT_PARAMETERS['SAMPLE_SIZE']
    recall_length = EXPERIMENT_PARAMETERS['RECALL_LENGTH']
    predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    sample_size_per_user = sample_size // num_user
    logger.info("Sample size per user: %s" % sample_size_per_user)
    X_all = []
    y_all = []
    for name, group in grouped:
        # logger.debug(name)
        # print(group)
        # group.reset_index()
        # print(group.iloc[0:1])
        # print(len(group))
        user_random_index = numpy.random.randint(low=0, high=(len(group) - (recall_length + predict_length)),size=sample_size_per_user)
        print(user_random_index)
        for i in numpy.nditer(user_random_index):
            i = int(i)
            x_user_all = group.iloc[i:i + recall_length, -1:].values.tolist()
            y_user_all = group.iloc[i + recall_length:i + recall_length + predict_length, -1:].values.tolist()
            X_all.append(x_user_all)
            if predict_length == 1:
                y_all.append(y_user_all[0])
            else:
                y_all.append(y_user_all)
    X_all = numpy.array(X_all)
    y_all = numpy.array(y_all)
    print('X_all shape:', X_all.shape)
    # print(X_all)
    print('y_all shape:', y_all.shape)
    # print(y_all)
    return X_all, y_all


def convert_dataframe_to_coordinate_and_grid_sequences(df: pandas.DataFrame, EXPERIMENT_PARAMETERS):
    # df = df.drop('timestamp', 1)
    # df = df.drop('t_index', 1)
    # df = df.drop('iso_year', 1)
    # df = df.drop('iso_week_number', 1)
    # df = df.drop('iso_weekday', 1)
    grouped = df.groupby('uid')
    num_user = len(grouped)
    sample_size = EXPERIMENT_PARAMETERS['SAMPLE_SIZE']
    prediction_input_length = EXPERIMENT_PARAMETERS['PREDICTION_INPUT_LENGTH']
    predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    sample_size_per_user = sample_size // num_user
    logger.info("Sample size per user: %s" % sample_size_per_user)
    X_all_coordinate = []
    y_all_coordinate = []
    X_all_grid = []
    y_all_grid = []

    for name, group in grouped:
        user_random_index = numpy.random.randint(low=0, high=(len(group) - (prediction_input_length + predict_length)), size=sample_size_per_user)
        for i in numpy.nditer(user_random_index):
            i = int(i)
            x_user_all_coordinate = group.iloc[i:i + prediction_input_length, -3:-1].values.tolist()
            y_user_all_coordinate = group.iloc[i + prediction_input_length:i + prediction_input_length + predict_length, -3:-1].values.tolist()
            x_user_all_grid = group.iloc[i:i + prediction_input_length, -1:].values.tolist()
            y_user_all_grid = group.iloc[i + prediction_input_length:i + prediction_input_length + predict_length, -1:].values.tolist()
            X_all_coordinate.append(x_user_all_coordinate)
            X_all_grid.append(x_user_all_grid)
            if predict_length == 1:
                y_all_coordinate.append(y_user_all_coordinate[0])
                y_all_grid.append(y_user_all_grid[0])
            else:
                y_all_coordinate.append(y_user_all_coordinate)
                y_all_grid.append(y_user_all_grid)
    X_all_coordinate = numpy.array(X_all_coordinate)
    y_all_coordinate = numpy.array(y_all_coordinate)
    X_all_grid = numpy.array(X_all_grid)
    y_all_grid = numpy.array(y_all_grid)
    print('X_all shape:', X_all_coordinate.shape)
    # print(X_all)
    print('y_all shape:', y_all_coordinate.shape)
    # print(y_all)
    return X_all_coordinate, y_all_coordinate, X_all_grid, y_all_grid


if __name__ == '__main__':
    slack_client = s.sc
    DATA_DIR = s.DATA_DIR
    GPS_FILTERED = s.GPS_FILTERED
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    X_COORDINATE_FILE = s.X_COORDINATE_FILE
    Y_COORDINATE_FILE = s.Y_COORDINATE_FILE
    X_GRID_FILE = s.X_GRID_FILE
    Y_GRID_FILE = s.Y_GRID_FILE
    TEMPORAL_DATAFRAME = s.TEMPORAL_DATAFRAME

    cores = mp.cpu_count()
    logger.info("Using %s cores" % cores)
    pool = mp.Pool(cores)

    temporal_index = define_temporal_index(EXPERIMENT_PARAMETERS)
    # # print(temporal_index)
    spatial_index = define_spatial_index(EXPERIMENT_PARAMETERS)
    # print(spatial_index)

    df_all_users = load_csv_files_to_dataframe(DATA_DIR, EXPERIMENT_PARAMETERS)
    df_user_filtered = filter_users_with_experiment_setting(df_all_users, EXPERIMENT_PARAMETERS)

    df_user_filtered.to_hdf(TEMPORAL_DATAFRAME, "df_all_users")
    df_user_filtered = pandas.read_hdf(TEMPORAL_DATAFRAME, "df_all_users")

    # Sequential
    # df_users_regularized = apply_slice_to_users(df_user_filtered, temporal_index, EXPERIMENT_PARAMETERS)
    # Parallel
    df_users_regularized = apply_slice_to_users_multiprocessing(df_user_filtered, temporal_index, spatial_index, EXPERIMENT_PARAMETERS, pool)

    df_users_regularized = filter_users_with_experiment_setting(df_users_regularized, EXPERIMENT_PARAMETERS)
    save_dataframe_to_csv(df_users_regularized, GPS_FILTERED, regularized=True)
    X_coordinate_all, y_coordinate_all, X_grid_all, y_grid_all  = convert_dataframe_to_coordinate_and_grid_sequences(df_users_regularized, EXPERIMENT_PARAMETERS)

    np.save(file=X_COORDINATE_FILE, arr=X_coordinate_all)
    np.save(file=Y_COORDINATE_FILE, arr=y_coordinate_all)
    np.save(file=X_GRID_FILE, arr=X_grid_all)
    np.save(file=Y_GRID_FILE, arr=y_grid_all)

    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__ + " is finished.")
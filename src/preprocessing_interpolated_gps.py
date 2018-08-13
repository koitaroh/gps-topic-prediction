import multiprocessing as mp
from itertools import repeat
import os
import random
import pandas as pd
from collections import OrderedDict
import numpy as np
from datetime import datetime, timedelta

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

import settings as s
import utility_io
import utility_spatiotemporal_index
import jpgrid
np.random.seed(1)
random.seed(1)


def filter_users_with_experiment_setting(DATAFRAME, EXPERIMENT_PARAMETERS):
    logger.info("Filtering dataframe with experiment parameters")
    df_users_filtered = DATAFRAME

    grouped = df_users_filtered.groupby('uid')
    logger.info("Number of uid: %s" % len(grouped))
    logger.info("Filtering with AOI")
    df_users_filtered = grouped.filter(lambda x: x['x'].min()>= float(EXPERIMENT_PARAMETERS["AOI"][0]))
    df_users_filtered = df_users_filtered.groupby('uid').filter(lambda x: x['x'].max()<= float(EXPERIMENT_PARAMETERS["AOI"][2]))
    df_users_filtered = df_users_filtered.groupby('uid').filter(lambda x: x['y'].min()>= float(EXPERIMENT_PARAMETERS["AOI"][1]))
    df_users_filtered = df_users_filtered.groupby('uid').filter(lambda x: x['y'].max()<= float(EXPERIMENT_PARAMETERS["AOI"][3]))

    grouped = df_users_filtered.groupby('uid')
    num_user = len(grouped)
    logger.info("Number of uid after spatial filter: %s" % num_user)

    if EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE'] <= num_user:
        num_user = EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE']
    sampled_df_i = random.sample(list(grouped.indices), num_user)
    df_list = map(lambda df_i: grouped.get_group(df_i), sampled_df_i)
    df_users_filtered = pd.concat(df_list, axis=0, join='outer')
    logger.info("Number of uid after sample user filter: %s" % len(df_users_filtered.groupby('uid')))
    return df_users_filtered


# apply slice to users
def apply_slice_to_users_parallel(df_users_filtered: pd.DataFrame, temporal_indices, EXPERIMENT_PARAMETERS, pool, topic_array):

    logger.info("Applying slice to users")

    # dtype = [('uid', 'int32'), ('timestamp', 'str'), ('t_index', 'int32'), ('y', 'float64'), ('x', 'float64'),
    #          ('mode', 'str'), ('s_index', 'int32')]

    # dtype = [
    #     ('uid', 'int32'), ('timestamp', 'str'), ('t_index', 'int32'), ('y', 'float64'), ('x', 'float64'),
    #     ('mode', 'str'), ('s_index', 'int32'),
    #     ("topic_1", 'float64'), ("topic_2", 'float64'), ("topic_3", 'float64'), ("topic_4", 'float64'),
    #     ("topic_5", 'float64'),
    #     ("topic_6", 'float64'), ("topic_7", 'float64'), ("topic_8", 'float64'), ("topic_9", 'float64'),
    #     ("topic_10", 'float64'),
    #     # ("topic_11", 'float64'),("topic_12", 'float64'), ("topic_13", 'float64'), ("topic_14", 'float64'), ("topic_15", 'float64'),
    #     # ("topic_16", 'float64'),("topic_17", 'float64'), ("topic_18", 'float64'), ("topic_19", 'float64'), ("topic_20", 'float64'),
    #     # ("topic_21", 'float64'),("topic_22", 'float64'), ("topic_23", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
    #     # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
    #     # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
    #     # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
    #     # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
    #     # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
    # ]

    # df_users_regularized = pandas.DataFrame(numpy.empty(0, dtype=dtype))
    # df_users_regularized = pandas.DataFrame(columns=headers)
    # print(df_users_filtered.dtypes)
    grouped = df_users_filtered.groupby('uid')
    df_list = pool.starmap(apply_t_index, zip(grouped, repeat(temporal_indices), repeat(EXPERIMENT_PARAMETERS), repeat(topic_array)))
    df_users_regularized = pd.concat(df_list)
    return df_users_regularized

def apply_t_index(grouped, temporal_index, EXPERIMENT_PARAMETERS, topic_array):
    name, group = grouped

    # dtype = [('uid', 'int32'), ('timestamp', 'str'), ('t_index', 'int32'), ('y', 'float64'), ('x', 'float64'),
    #          ('mode', 'str'), ('s_index', 'int32')]

    dtype = [
        ('uid', 'int32'), ('timestamp', 'str'), ('t_index', 'int32'), ('y', 'float64'), ('x', 'float64'),
        ('mode', 'str'), ('s_index', 'str'),
        ("topic_0", 'float64'),("topic_1", 'float64'), ("topic_2", 'float64'), ("topic_3", 'float64'), ("topic_4", 'float64'),
        ("topic_5", 'float64'),("topic_6", 'float64'), ("topic_7", 'float64'), ("topic_8", 'float64'), ("topic_9", 'float64'),
        # ("topic_11", 'float64'),("topic_12", 'float64'), ("topic_13", 'float64'), ("topic_14", 'float64'), ("topic_15", 'float64'),
        # ("topic_16", 'float64'),("topic_17", 'float64'), ("topic_18", 'float64'), ("topic_19", 'float64'), ("topic_20", 'float64'),
        # ("topic_21", 'float64'),("topic_22", 'float64'), ("topic_23", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
        # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
        # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
        # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
        # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
        # ("topic_1", 'float64'),("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'), ("topic_1", 'float64'),
    ]
    df_user_regularized = pd.DataFrame(np.empty(0, dtype=dtype))
    for (i, t_index) in enumerate(temporal_index):
        target_time = t_index[4]
        target_row = group.loc[(group['time_start'] <= target_time) & (group['time_end'] >= target_time)]
        if (len(target_row) > 0):
            uid = target_row.iat[0, 0]
            x = target_row.iat[0, 3]
            y = target_row.iat[0, 4]
            mode = target_row.iat[0, 5]
            s_index = jpgrid.encodeBase(y, x)
            topic_vector = topic_array[i, :].tolist()
            df_user_regularized_one = pd.DataFrame(OrderedDict((
                    ('uid', [uid]), ("timestamp", [target_time]), ("t_index", [t_index[0]]), ("y", [y]), ("x", [x]), ("mode", [mode]), ("s_index", [s_index]),
                    ("topic_0", [topic_vector[0]]), ("topic_1", [topic_vector[1]]), ("topic_2", [topic_vector[2]]), ("topic_3", [topic_vector[3]]), ("topic_4", [topic_vector[4]]),
                    ("topic_5", [topic_vector[5]]), ("topic_6", [topic_vector[6]]), ("topic_7", [topic_vector[7]]), ("topic_8", [topic_vector[8]]), ("topic_9", [topic_vector[9]]),
            )))
            # df_user_regularized_one = pandas.DataFrame(OrderedDict((('uid', [uid]), ("timestamp", [target_time]), ("t_index", [t_index[0]]), ("y", [y]), ("x", [x]), ("mode", [mode]), ("s_index", [s_index]))))
            df_user_regularized = df_user_regularized.append(df_user_regularized_one)[df_user_regularized_one.columns.tolist()]
    # print(df_user_regularized)
    return df_user_regularized


def convert_dataframe_to_coordinate_mode_and_grid_mode_topic_sequences(df: pd.DataFrame, EXPERIMENT_PARAMETERS):
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
    X_all_mode = []
    y_all_mode = []
    X_all_topic = []
    y_all_topic = []

    X_evaluation_coordinate = []
    y_evaluation_coordinate = []
    X_evaluation_grid = []
    y_evaluation_grid = []
    X_evaluation_mode = []
    y_evaluation_mode = []
    X_evaluation_topic = []
    y_evaluation_topic = []

    for name, group in grouped:
        if (len(group) - (prediction_input_length + predict_length)) > 0:
            user_random_index = np.random.randint(low=0, high=(len(group) - (prediction_input_length + predict_length)), size=sample_size_per_user)
            for i in np.nditer(user_random_index):
                i = int(i)
                x_user_all_coordinate = group.iloc[i:i + prediction_input_length, 3:5].values.tolist()
                y_user_all_coordinate = group.iloc[i + prediction_input_length:i + prediction_input_length + predict_length, 3:5].values.tolist()
                x_user_all_grid = group.iloc[i:i + prediction_input_length, 6:7].values.tolist()
                y_user_all_grid = group.iloc[i + prediction_input_length:i + prediction_input_length + predict_length, 6:7].values.tolist()
                x_user_all_mode = group.iloc[i:i + prediction_input_length, 5:6].values.tolist()
                y_user_all_mode = group.iloc[i + prediction_input_length:i + prediction_input_length + predict_length, 5:6].values.tolist()
                x_user_all_topic = group.iloc[i:i + prediction_input_length, 7:17].values.tolist()
                y_user_all_topic = group.iloc[i + prediction_input_length:i + prediction_input_length + predict_length, 7:17].values.tolist()
                X_all_coordinate.append(x_user_all_coordinate)
                X_all_grid.append(x_user_all_grid)
                X_all_mode.append(x_user_all_mode)
                X_all_topic.append(x_user_all_topic)
                if predict_length == 1:
                    y_all_coordinate.append(y_user_all_coordinate[0])
                    y_all_grid.append(y_user_all_grid[0])
                    y_all_mode.append(y_user_all_mode[0])
                    y_all_topic.append(y_user_all_topic[0])
                else:
                    y_all_coordinate.append(y_user_all_coordinate)
                    y_all_grid.append(y_user_all_grid)
                    y_all_mode.append(y_user_all_mode)
                    y_all_topic.append(y_user_all_topic)
    X_all_coordinate = np.array(X_all_coordinate)
    y_all_coordinate = np.array(y_all_coordinate)
    X_all_grid = np.array(X_all_grid)
    y_all_grid = np.array(y_all_grid)
    X_all_mode = np.array(X_all_mode)
    y_all_mode = np.array(y_all_mode)
    X_all_topic = np.array(X_all_topic)
    y_all_topic = np.array(y_all_topic)
    print('X_all_coordinate shape:', X_all_coordinate.shape)
    print('y_all_coordinate shape:', y_all_coordinate.shape)
    print('X_all_grid shape:', X_all_grid.shape)
    print('y_all_grid shape:', y_all_grid.shape)
    print('X_all_mode shape:', X_all_mode.shape)
    print('y_all_mode shape:', y_all_mode.shape)
    print('X_all_topic shape:', X_all_topic.shape)
    print('y_all_topic shape:', y_all_topic.shape)
    return X_all_coordinate, y_all_coordinate, X_all_grid, y_all_grid, X_all_mode, y_all_mode, X_all_topic, y_all_topic


if __name__ == '__main__':
    slack_client = s.sc
    GPS_RAW_DIR = s.GPS_RAW_DIR
    GPS_INTERPOLATED_FILTERED = s.GPS_INTERPOLATED_FILTERED
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS

    X_COORDINATE_FILE = s.X_COORDINATE_FILE
    Y_COORDINATE_FILE = s.Y_COORDINATE_FILE
    X_GRID_FILE = s.X_GRID_FILE
    X_MODE_FILE = s.X_MODE_FILE
    Y_GRID_FILE = s.Y_GRID_FILE
    Y_MODE_FILE = s.Y_MODE_FILE
    X_TOPIC_FILE = s.X_TOPIC_FILE
    Y_TOPIC_FILE = s.Y_TOPIC_FILE

    TEMPORAL_DATAFRAME = s.TEMPORAL_DATAFRAME
    LSI_TOPIC_FILE = s.LSI_TOPIC_FILE

    cores = mp.cpu_count()
    logger.info("Using %s cores" % cores)
    pool = mp.Pool(cores)

    temporal_index = utility_spatiotemporal_index.define_temporal_index(EXPERIMENT_PARAMETERS)
    spatial_index = utility_spatiotemporal_index.define_spatial_index(EXPERIMENT_PARAMETERS)
    topic_array = np.load(LSI_TOPIC_FILE)

    # Load CSV files
    df_all_users = utility_io.load_csv_files_to_dataframe(GPS_RAW_DIR, EXPERIMENT_PARAMETERS)
    df_user_filtered = filter_users_with_experiment_setting(df_all_users, EXPERIMENT_PARAMETERS)
    # df_user_filtered.to_hdf(TEMPORAL_DATAFRAME, "df_all_users")

    # Load dataframe
    # df_user_filtered = pd.read_hdf(TEMPORAL_DATAFRAME, "df_all_users")
    df_users_regularized = apply_slice_to_users_parallel(df_user_filtered, temporal_index, EXPERIMENT_PARAMETERS, pool, topic_array)
    df_users_regularized.to_hdf(TEMPORAL_DATAFRAME, "df_users_regularized")
    utility_io.save_dataframe_to_csv(df_users_regularized, GPS_INTERPOLATED_FILTERED, regularized=True)

    # Load Reguralized dataframe
    df_users_regularized = pd.read_hdf(TEMPORAL_DATAFRAME, "df_users_regularized")
    # print(df_users_regularized)
    X_coordinate_all, y_coordinate_all, X_grid_all, y_grid_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all  = convert_dataframe_to_coordinate_mode_and_grid_mode_topic_sequences(df_users_regularized, EXPERIMENT_PARAMETERS)

    # print(X_coordinate_all[0])
    # print(y_coordinate_all[0])
    # print(X_grid_all[0])
    # print(y_grid_all[0])
    # print(X_mode_all[0])
    # print(X_topic_all[0])
    # print(y_topic_all[0])

    np.save(file=X_COORDINATE_FILE, arr=X_coordinate_all)
    np.save(file=Y_COORDINATE_FILE, arr=y_coordinate_all)
    np.save(file=X_GRID_FILE, arr=X_grid_all)
    np.save(file=Y_GRID_FILE, arr=y_grid_all)
    np.save(file=X_MODE_FILE, arr=X_mode_all)
    np.save(file=Y_MODE_FILE, arr=y_mode_all)
    np.save(file=X_TOPIC_FILE, arr=X_topic_all)
    np.save(file=Y_TOPIC_FILE, arr=y_topic_all)

    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__ + " is finished.")
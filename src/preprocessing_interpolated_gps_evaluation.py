import csv
import os
import random
import multiprocessing as mp
from itertools import repeat
import pandas as pd
from collections import OrderedDict
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import geojson
from geojson import LineString, FeatureCollection, Feature

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


def convert_dataframe_to_coordinate_mode_and_grid_mode_topic_sequences_evaluation(df: pd.DataFrame, EXPERIMENT_PARAMETERS):
    grouped = df.groupby('uid')
    num_user = len(grouped)
    sample_size = EXPERIMENT_PARAMETERS['EVALUATION_SAMPLE_SIZE']
    prediction_input_length = EXPERIMENT_PARAMETERS['PREDICTION_INPUT_LENGTH']
    prediction_output_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    logger.info("Sample size per user: %s" % sample_size)
    logger.info("Prediction input length: %s" % prediction_input_length)
    logger.info("Prediction output length: %s" % prediction_output_length)
    X_all_coordinate = []
    y_all_coordinate = []
    X_all_grid = []
    y_all_grid = []
    X_all_mode = []
    y_all_mode = []
    X_all_topic = []
    y_all_topic = []
    for name, group in grouped:
        if len(group) == (prediction_input_length + prediction_output_length):
            x_user_all_coordinate = group.iloc[0:prediction_input_length, 3:5].values.tolist()
            y_user_all_coordinate = group.iloc[prediction_input_length:prediction_input_length + prediction_output_length, 3:5].values.tolist()
            x_user_all_grid = group.iloc[0:prediction_input_length, 6:7].values.tolist()
            y_user_all_grid = group.iloc[prediction_input_length:prediction_input_length + prediction_output_length, 6:7].values.tolist()
            x_user_all_mode = group.iloc[0:prediction_input_length, 5:6].values.tolist()
            y_user_all_mode = group.iloc[prediction_input_length:prediction_input_length + prediction_output_length, 5:6].values.tolist()
            x_user_all_topic = group.iloc[0:prediction_input_length, 7:17].values.tolist()
            y_user_all_topic = group.iloc[prediction_input_length:prediction_input_length + prediction_output_length, 7:17].values.tolist()
            X_all_coordinate.append(x_user_all_coordinate)
            X_all_grid.append(x_user_all_grid)
            X_all_mode.append(x_user_all_mode)
            X_all_topic.append(x_user_all_topic)
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


def create_geojson_line_from_dataframe(df, outfile, EXPERIMENT_PARAMETERS):
    logger.info("Creating GeoJSON file")

    predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    with open(outfile, 'w', encoding="utf-8") as geojson_file:
        my_feature_list = []
        new_uid = 0
        grouped = df.groupby('uid')
        for name, group in tqdm(grouped):
            properties = {"new_uid": new_uid}
            line = []
            for index, row in group.iterrows():
                lat, lon = row['y'], row['x']
                line.append((lon, lat))
            new_uid += 1
            my_line = LineString(line)
            my_feature = Feature(geometry=my_line, properties=properties)
            my_feature_list.append(my_feature)
        my_feature_collection = FeatureCollection(my_feature_list)
        dump = geojson.dumps(my_feature_collection, sort_keys=True)
        geojson_file.write(dump)
    return None


# def create_csv_trajectory_from_dataframe(df, csv_file, temporal_index, EXPERIMENT_PARAMETERS):
#     logger.info("Saving trajectory to CSV %s" % csv_file)
#     with open(csv_file, 'w') as f:
#         writer = csv.writer(f)
#         X_shape = X.shape
#         # print(X_shape) # (50, 144, 2)
#         X_list = X.tolist()
#         print(X_list)
#         for i in tqdm(range(len(X_list))):
#             for j in tqdm(range(len(X_list[0]))):
#                 id = i
#                 t = temporal_index[j][4].strftime('%Y-%m-%d %H:%M:%S')
#                 x = X_list[i][j][1]
#                 y = X_list[i][j][0]
#                 writer.writerow([id, t, x, y])
#     return True


if __name__ == '__main__':
    slack_client = s.sc
    GPS_RAW_DIR = s.GPS_RAW_DIR
    GPS_INTERPOLATED_FILTERED = s.GPS_INTERPOLATED_FILTERED
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS

    X_COORDINATE_EVALUATION_FILE = s.X_COORDINATE_EVALUATION_FILE
    Y_COORDINATE_EVALUATION_FILE = s.Y_COORDINATE_EVALUATION_FILE
    X_MODE_EVALUATION_FILE = s.X_MODE_EVALUATION_FILE
    X_GRID_EVALUATION_FILE = s.X_GRID_EVALUATION_FILE
    Y_GRID_EVALUATION_FILE = s.Y_GRID_EVALUATION_FILE
    Y_MODE_EVALUATION_FILE = s.Y_MODE_EVALUATION_FILE
    X_TOPIC_EVALUATION_FILE = s.X_TOPIC_EVALUATION_FILE
    Y_TOPIC_EVALUATION_FILE = s.Y_TOPIC_EVALUATION_FILE

    TEMPORAL_DATAFRAME = s.TEMPORAL_DATAFRAME
    LSI_TOPIC_EVALUATION_FILE = s.LSI_TOPIC_EVALUATION_FILE
    LDA_TOPIC_EVALUATION_FILE = s.LDA_TOPIC_EVALUATION_FILE
    DOC2VEC_TOPIC_EVALUATION_FILE = s.DOC2VEC_TOPIC_EVALUATION_FILE

    # GEOJSON_FILE_EVALUATION_RAW = s.GEOJSON_FILE_EVALUATION_RAW
    CSV_TRAJECTORY_FILE_EVALUATION_RAW = s.CSV_TRAJECTORY_FILE_EVALUATION_RAW

    cores = mp.cpu_count()
    logger.info("Using {} cores".format(cores))
    pool = mp.Pool(cores)

    temporal_index = utility_spatiotemporal_index.define_temporal_index_evaluation(EXPERIMENT_PARAMETERS)
    topic_array = np.load(LDA_TOPIC_EVALUATION_FILE)

    df_all_users = utility_io.load_csv_files_to_dataframe_evaluation(GPS_RAW_DIR, EXPERIMENT_PARAMETERS)
    df_user_filtered = filter_users_with_experiment_setting(df_all_users, EXPERIMENT_PARAMETERS)

    df_user_filtered.to_hdf(TEMPORAL_DATAFRAME, "df_all_users")
    df_user_filtered = pd.read_hdf(TEMPORAL_DATAFRAME, "df_all_users")

    df_users_regularized = apply_slice_to_users_parallel(df_user_filtered, temporal_index, EXPERIMENT_PARAMETERS, pool, topic_array)

    df_users_regularized.to_hdf(TEMPORAL_DATAFRAME, "df_all_users")
    df_users_regularized = pd.read_hdf(TEMPORAL_DATAFRAME, "df_all_users")
    # print(df_users_regularized.head)

    # create_geojson_line_from_dataframe(df_users_regularized, GEOJSON_FILE_EVALUATION_RAW, EXPERIMENT_PARAMETERS)

    # save_dataframe_to_csv(df_users_regularized, GPS_INTERPOLATED_FILTERED, regularized=True)

    X_coordinate_all, y_coordinate_all, X_grid_all, y_grid_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all  = convert_dataframe_to_coordinate_mode_and_grid_mode_topic_sequences_evaluation(df_users_regularized, EXPERIMENT_PARAMETERS)

    # print(X_coordinate_all[0])
    # print(y_coordinate_all[0])
    # print(X_grid_all[0])
    # print(y_grid_all[0])
    # print(X_mode_all[0])
    # print(X_topic_all[0])
    # print(y_topic_all[0])

    np.save(file=X_COORDINATE_EVALUATION_FILE, arr=X_coordinate_all)
    np.save(file=Y_COORDINATE_EVALUATION_FILE, arr=y_coordinate_all)
    np.save(file=X_GRID_EVALUATION_FILE, arr=X_grid_all)
    np.save(file=Y_GRID_EVALUATION_FILE, arr=y_grid_all)
    np.save(file=X_MODE_EVALUATION_FILE, arr=X_mode_all)
    np.save(file=Y_MODE_EVALUATION_FILE, arr=y_mode_all)
    np.save(file=X_TOPIC_EVALUATION_FILE, arr=X_topic_all)
    np.save(file=Y_TOPIC_EVALUATION_FILE, arr=y_topic_all)

    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__ + " is finished.")
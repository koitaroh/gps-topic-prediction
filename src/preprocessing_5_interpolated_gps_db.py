import multiprocessing as mp
from itertools import repeat
import os
import warnings
import random
import atexit
import pandas as pd
from collections import OrderedDict
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from sqlalchemy import create_engine, text


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
start_time = datetime.now()

import settings as s
import utility_io
import utility_database
import utility_spatiotemporal_index
import jpgrid
np.random.seed(1)
random.seed(1)


def load_gps_trajectory_db_to_dataframe(GPS_TABLE_NAME, PROFILE_TABLE_NAME, EXPERIMENT_PARAMETERS, conn, temporal_index, topic_array):
    prediction_input_length = EXPERIMENT_PARAMETERS['PREDICTION_INPUT_LENGTH']
    predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    sample_user_size = EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE']
    sql = f"""select uid from {PROFILE_TABLE_NAME} where preprocessing is NULL limit {sample_user_size}"""
    profile_df = pd.read_sql_query(text(sql), conn)
    # print(profile_df)

    X_all_coordinate = []
    y_all_coordinate = []
    X_all_grid = []
    y_all_grid = []
    X_all_mode = []
    y_all_mode = []
    X_all_topic = []
    y_all_topic = []

    for i, row in tqdm(enumerate(profile_df.itertuples(), 1)):
        # print(i, row.uid)
        user_trajectory_sql = f"""select uid, timestamp_from, timestamp_to, longitude, latitude, mode from gps_interpolated where uid = {row.uid}"""
        user_trajectory_df = pd.read_sql_query(text(user_trajectory_sql), conn)
        user_trajectory_regularized_df = apply_stt_labels(user_trajectory_df, temporal_index, topic_array, EXPERIMENT_PARAMETERS)
        user_trajectory_regularized_df.to_sql(name=GPS_TABLE_NAME, con=conn, if_exists='append')

        x_user_all_coordinate = user_trajectory_regularized_df.iloc[0:prediction_input_length, 3:5].values.tolist()
        y_user_all_coordinate = user_trajectory_regularized_df.iloc[prediction_input_length:prediction_input_length + predict_length,3:5].values.tolist()
        x_user_all_grid = user_trajectory_regularized_df.iloc[0:prediction_input_length, 6:7].values.tolist()
        y_user_all_grid = user_trajectory_regularized_df.iloc[prediction_input_length:prediction_input_length + predict_length,6:7].values.tolist()
        x_user_all_mode = user_trajectory_regularized_df.iloc[0:prediction_input_length, 5:6].values.tolist()
        y_user_all_mode = user_trajectory_regularized_df.iloc[prediction_input_length:prediction_input_length + predict_length,5:6].values.tolist()
        x_user_all_topic = user_trajectory_regularized_df.iloc[0:prediction_input_length, 7:17].values.tolist()
        y_user_all_topic = user_trajectory_regularized_df.iloc[prediction_input_length:prediction_input_length + predict_length,7:17].values.tolist()

        assert len(x_user_all_coordinate) == prediction_input_length, f'uid: {row.uid}. prediction_input_length: {prediction_input_length}, len(x_user_all_coordinate): {len(x_user_all_coordinate)}'
        assert len(y_user_all_coordinate) == predict_length, f'uid: {row.uid}. predict_length: {predict_length}, len(y_user_all_coordinate): {len(y_user_all_coordinate)}'

        X_all_coordinate.append(x_user_all_coordinate)
        X_all_grid.append(x_user_all_grid)
        X_all_mode.append(x_user_all_mode)
        X_all_topic.append(x_user_all_topic)
        y_all_coordinate.append(y_user_all_coordinate)
        y_all_grid.append(y_user_all_grid)
        y_all_mode.append(y_user_all_mode)
        y_all_topic.append(y_user_all_topic)

    X_all_coordinate = np.array(X_all_coordinate, dtype=np.float32)
    y_all_coordinate = np.array(y_all_coordinate, dtype=np.float32)
    X_all_grid = np.asarray(X_all_grid)
    y_all_grid = np.asarray(y_all_grid)
    X_all_mode = np.asarray(X_all_mode)
    y_all_mode = np.asarray(y_all_mode)
    X_all_topic = np.asarray(X_all_topic, dtype=np.float32)
    y_all_topic = np.asarray(y_all_topic, dtype=np.float32)
    logger.info(f'X_all_coordinate shape: {X_all_coordinate.shape} dtype: {X_all_coordinate.dtype}')
    logger.info(f'y_all_coordinate shape: {y_all_coordinate.shape} dtype: {y_all_coordinate.dtype}')
    logger.info(f'X_all_grid shape: {X_all_grid.shape} dtype: {X_all_grid.dtype}')
    logger.info(f'y_all_grid shape: {y_all_grid.shape} dtype: {y_all_grid.dtype}')
    logger.info(f'X_all_mode shape: {X_all_mode.shape} dtype: {X_all_mode.dtype}')
    logger.info(f'y_all_mode shape: {y_all_mode.shape} dtype: {y_all_mode.dtype}')
    logger.info(f'X_all_topic shape: {X_all_topic.shape} dtype: {X_all_topic.dtype}')
    logger.info(f'y_all_topic shape: {y_all_topic.shape} dtype: {y_all_topic.dtype}')
    return X_all_coordinate, y_all_coordinate, X_all_grid, y_all_grid, X_all_mode, y_all_mode, X_all_topic, y_all_topic


def apply_stt_labels(user_trajectory_df, temporal_index, topic_array, EXPERIMENT_PARAMETERS):
    dtype = [
        ('uid', 'int32'), ('timestamp', 'str'), ('t_index', 'int32'), ('y', 'float64'), ('x', 'float64'),
        ('mode', 'str'), ('s_index', 'str'),
        ("topic_0", 'float64'), ("topic_1", 'float64'), ("topic_2", 'float64'), ("topic_3", 'float64'),
        ("topic_4", 'float64'),
        ("topic_5", 'float64'), ("topic_6", 'float64'), ("topic_7", 'float64'), ("topic_8", 'float64'),
        ("topic_9", 'float64'),
        ]
    user_trajectory_regularized_df = pd.DataFrame(np.empty(0, dtype=dtype))

    for (i, t_index) in enumerate(temporal_index):
        target_time = t_index[4]
        # target_row = user_trajectory_df.loc[(user_trajectory_df['timestamp_from'] <= target_time) & (user_trajectory_df['timestamp_to'] >= target_time)]

        target_row = user_trajectory_df.iloc[user_trajectory_df.index.get_loc(target_time, method='nearest')]

        if (len(target_row) > 0):
            uid = target_row.iat[0, 0]
            x = target_row.iat[0, 3]
            y = target_row.iat[0, 4]
            mode = target_row.iat[0, 5]
            s_index = jpgrid.encodeBase(y, x)
            topic_vector = topic_array[i, :].tolist()
            user_trajectory_regularized_t_one_df = pd.DataFrame(OrderedDict((
                ('uid', [uid]), ("timestamp", [target_time]), ("t_index", [t_index[0]]), ("y", [y]), ("x", [x]),
                ("mode", [mode]), ("s_index", [s_index]),
                ("topic_0", [topic_vector[0]]), ("topic_1", [topic_vector[1]]), ("topic_2", [topic_vector[2]]),
                ("topic_3", [topic_vector[3]]), ("topic_4", [topic_vector[4]]),
                ("topic_5", [topic_vector[5]]), ("topic_6", [topic_vector[6]]), ("topic_7", [topic_vector[7]]),
                ("topic_8", [topic_vector[8]]), ("topic_9", [topic_vector[9]]),
            )))
            user_trajectory_regularized_df = user_trajectory_regularized_df.append(user_trajectory_regularized_t_one_df)[user_trajectory_regularized_t_one_df.columns.tolist()]

    return user_trajectory_regularized_df


def load_gps_trajectory_db_to_dataframe_parallel(GPS_TABLE_RAW_NAME, GPS_TABLE_REGULARIZED_NAME, PROFILE_TABLE_NAME, EXPERIMENT_PARAMETERS, conn, temporal_index, topic_array):
    sample_user_size = EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE']
    sql = f"""
            select uid from {PROFILE_TABLE_NAME} 
            where preprocessing is NULL 
--             and pred_s_jpgrid != pred_e_jpgrid
            limit {sample_user_size}
        """
    profile_df = pd.read_sql_query(text(sql), conn)
    uid_list = profile_df['uid'].tolist()
    chunk_size = 20000
    df_chunk_list = []
    for i in tqdm(range(0, len(uid_list), chunk_size)):
        uid_list_sub = uid_list[i:i+chunk_size]

        uid_tuple = tuple(uid_list_sub)
        users_trajectory_sql = f"""
                    SELECT uid, timestamp_from, latitude, longitude, mode
                    FROM gps_interpolated
                    where uid in {uid_tuple}
                    and (latitude between {EXPERIMENT_PARAMETERS['AOI'][1]} and {EXPERIMENT_PARAMETERS['AOI'][3]})
                    and (longitude between {EXPERIMENT_PARAMETERS['AOI'][0]} and {EXPERIMENT_PARAMETERS['AOI'][2]})
                    and timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}'
                """
        users_trajectory_df = pd.read_sql_query(text(users_trajectory_sql), conn, parse_dates=['timestamp_from'])
        grouped = users_trajectory_df.groupby('uid', as_index=False)
        num_user = len(grouped)
        # logger.info("Number of uid after spatial filter: %s" % num_user)
        assert num_user > 0, 'Number of uid is zero.'
        df_list = pool.starmap(apply_stt_labels_parallel, zip(grouped, repeat(temporal_index), repeat(topic_array), repeat(EXPERIMENT_PARAMETERS)))
        df_users_regularized = pd.concat(df_list)
        if i == 0:
            users_trajectory_df.to_sql(name=GPS_TABLE_RAW_NAME, con=conn, if_exists='replace')
            df_users_regularized.to_sql(name=GPS_TABLE_REGULARIZED_NAME, con=conn, if_exists='replace')
        else:
            users_trajectory_df.to_sql(name=GPS_TABLE_RAW_NAME, con=conn, if_exists='append')
            df_users_regularized.to_sql(name=GPS_TABLE_REGULARIZED_NAME, con=conn, if_exists='append')
        df_chunk_list.append(df_users_regularized)
    df_users_chunk_regularized = pd.concat(df_chunk_list)
    logger.info(f"Concatenating interpolated dataframes of len: {len(df_users_chunk_regularized)}")
    return df_users_chunk_regularized


def load_gps_trajectory_db_to_dataframe_parallel_evaluation(GPS_TABLE_RAW_NAME, GPS_TABLE_REGULARIZED_NAME, PROFILE_TABLE_NAME, EXPERIMENT_PARAMETERS, conn, temporal_index, topic_array):
    sample_user_size = EXPERIMENT_PARAMETERS['EVALUATION_SAMPLE_SIZE']
    sql = f"""
            select uid from {PROFILE_TABLE_NAME} 
            where preprocessing is NULL 
            and pred_s_jpgrid != pred_e_jpgrid
            limit {sample_user_size}
        """
    profile_df = pd.read_sql_query(text(sql), conn)
    uid_list = profile_df['uid'].tolist()
    chunk_size = 20000
    df_chunk_list = []
    for i in tqdm(range(0, len(uid_list), chunk_size)):
        uid_list_sub = uid_list[i:i+chunk_size]

        uid_tuple = tuple(uid_list_sub)
        users_trajectory_sql = f"""
                    SELECT uid, timestamp_from, latitude, longitude, mode
                    FROM gps_interpolated
                    where uid in {uid_tuple}
                    and (latitude between {EXPERIMENT_PARAMETERS['AOI'][1]} and {EXPERIMENT_PARAMETERS['AOI'][3]})
                    and (longitude between {EXPERIMENT_PARAMETERS['AOI'][0]} and {EXPERIMENT_PARAMETERS['AOI'][2]})
                    and timestamp_from between '{EXPERIMENT_PARAMETERS['EVALUATION_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['EVALUATION_PREDICTION_END']}'
                """
        users_trajectory_df = pd.read_sql_query(text(users_trajectory_sql), conn, parse_dates=['timestamp_from'])
        grouped = users_trajectory_df.groupby('uid', as_index=False)
        num_user = len(grouped)
        # logger.info("Number of uid after spatial filter: %s" % num_user)
        assert num_user > 0, 'Number of uid is zero.'
        df_list = pool.starmap(apply_stt_labels_parallel, zip(grouped, repeat(temporal_index), repeat(topic_array), repeat(EXPERIMENT_PARAMETERS)))
        df_users_regularized = pd.concat(df_list)
        # if i == 0:
        #     users_trajectory_df.to_sql(name=GPS_TABLE_RAW_NAME, con=conn, if_exists='replace')
        #     df_users_regularized.to_sql(name=GPS_TABLE_REGULARIZED_NAME, con=conn, if_exists='replace')
        # else:
        #     users_trajectory_df.to_sql(name=GPS_TABLE_RAW_NAME, con=conn, if_exists='append')
        #     df_users_regularized.to_sql(name=GPS_TABLE_REGULARIZED_NAME, con=conn, if_exists='append')
        df_chunk_list.append(df_users_regularized)
    df_users_chunk_regularized = pd.concat(df_chunk_list)
    logger.info(f"Concatenating interpolated dataframes of len: {len(df_users_chunk_regularized)}")
    return df_users_chunk_regularized


def apply_stt_labels_parallel(grouped, temporal_index, topic_array, EXPERIMENT_PARAMETERS):
    name, group = grouped
    group = group.drop_duplicates('timestamp_from')
    group.index = pd.to_datetime(group['timestamp_from'])
    group = group.sort_index()

    # dtype = [('uid', 'int32'), ('timestamp', 'str'), ('t_index', 'int32'), ('y', 'float64'), ('x', 'float64'),
    #          ('mode', 'str'), ('s_index', 'int32')]

    dtype = [
        ('uid', 'int32'), ('timestamp', 'str'), ('t_index', 'int32'), ('y', 'float64'), ('x', 'float64'),
        ('mode', 'str'), ('s_index', 'str'),
        ("topic_0", 'float64'), ("topic_1", 'float64'), ("topic_2", 'float64'), ("topic_3", 'float64'),
        ("topic_4", 'float64'),
        ("topic_5", 'float64'), ("topic_6", 'float64'), ("topic_7", 'float64'), ("topic_8", 'float64'),
        ("topic_9", 'float64'),
    ]
    df_user_regularized = pd.DataFrame(np.empty(0, dtype=dtype))
    for (i, t_index) in enumerate(temporal_index):
        target_time = t_index[4]
        # logger.info(target_time)
        # target_row = group.iloc[group.index.get_loc(target_time, method='nearest')]
        target_row = group.iloc[group.index.get_indexer([target_time], method='nearest')[0]]

        # target_row = group.loc[(group['time_start'] <= target_time) & (group['time_end'] >= target_time)]
        if (len(target_row) > 0):
            uid = target_row.iat[0]
            x = target_row.iat[3]
            y = target_row.iat[2]
            mode = target_row.iat[4]
            s_index = jpgrid.encodeBase(y, x)
            topic_vector = topic_array[i, :].tolist()
            user_trajectory_regularized_t_one_df = pd.DataFrame(OrderedDict((
                ('uid', [uid]), ("timestamp", [target_time]), ("t_index", [t_index[0]]), ("y", [y]), ("x", [x]),
                ("mode", [mode]), ("s_index", [s_index]),
                ("topic_0", [topic_vector[0]]), ("topic_1", [topic_vector[1]]), ("topic_2", [topic_vector[2]]),
                ("topic_3", [topic_vector[3]]), ("topic_4", [topic_vector[4]]),
                ("topic_5", [topic_vector[5]]), ("topic_6", [topic_vector[6]]), ("topic_7", [topic_vector[7]]),
                ("topic_8", [topic_vector[8]]), ("topic_9", [topic_vector[9]]),
            )))
            # df_user_regularized_one = pandas.DataFrame(OrderedDict((('uid', [uid]), ("timestamp", [target_time]), ("t_index", [t_index[0]]), ("y", [y]), ("x", [x]), ("mode", [mode]), ("s_index", [s_index]))))
            # df_user_regularized = df_user_regularized.append(user_trajectory_regularized_t_one_df)[user_trajectory_regularized_t_one_df.columns.tolist()]
            df_user_regularized = pd.concat([df_user_regularized, user_trajectory_regularized_t_one_df], ignore_index=True)

    # print(df_user_regularized)
    return df_user_regularized


def convert_dataframe_to_coordinate_mode_and_grid_mode_topic_sequences(df: pd.DataFrame, EXPERIMENT_PARAMETERS):
    prediction_input_length = EXPERIMENT_PARAMETERS['PREDICTION_INPUT_LENGTH']
    predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    X_all_coordinate = []
    y_all_coordinate = []
    X_all_grid = []
    y_all_grid = []
    X_all_mode = []
    y_all_mode = []
    X_all_topic = []
    y_all_topic = []
    grouped = df.groupby('uid')
    for name, group in grouped:
        x_user_all_coordinate = group.iloc[0:prediction_input_length, 3:5].values.tolist()
        y_user_all_coordinate = group.iloc[prediction_input_length:prediction_input_length + predict_length, 3:5].values.tolist()
        x_user_all_grid = group.iloc[0:prediction_input_length, 6:7].values.tolist()
        y_user_all_grid = group.iloc[prediction_input_length:prediction_input_length + predict_length,6:7].values.tolist()
        x_user_all_mode = group.iloc[0:prediction_input_length, 5:6].values.tolist()
        y_user_all_mode = group.iloc[prediction_input_length:prediction_input_length + predict_length,5:6].values.tolist()
        x_user_all_topic = group.iloc[0:prediction_input_length, 7:17].values.tolist()
        y_user_all_topic = group.iloc[prediction_input_length:prediction_input_length + predict_length,7:17].values.tolist()
        assert len(x_user_all_coordinate) == prediction_input_length, f'uid: {name}. prediction_input_length: {prediction_input_length}, len(x_user_all_coordinate): {len(x_user_all_coordinate)}'
        assert len(y_user_all_coordinate) == predict_length, f'uid: {name}. predict_length: {predict_length}, len(y_user_all_coordinate): {len(y_user_all_coordinate)}'

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


if __name__ == '__main__':
    slack_client = s.slack_client
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    EXPERIMENT_ENVIRONMENT = s.EXPERIMENT_ENVIRONMENT
    SCENARIO = s.SCENARIO
    PROFILE_TABLE_NAME = s.PROFILE_TABLE_NAME # original
    # PROFILE_TABLE_NAME = s.PROFILE_TABLE_NAME_SMALL_TARGET # changed
    PROFILE_TABLE_NAME_SMALL_TARGET = s.PROFILE_TABLE_NAME_SMALL_TARGET
    GPS_TABLE_RAW_NAME = s.GPS_TABLE_RAW_NAME
    GPS_TABLE_REGULARIZED_NAME = s.GPS_TABLE_REGULARIZED_NAME

    X_COORDINATE_FILE = s.X_COORDINATE_FILE
    Y_COORDINATE_FILE = s.Y_COORDINATE_FILE
    X_GRID_FILE = s.X_GRID_FILE
    X_MODE_FILE = s.X_MODE_FILE
    Y_GRID_FILE = s.Y_GRID_FILE
    Y_MODE_FILE = s.Y_MODE_FILE
    X_TOPIC_FILE = s.X_TOPIC_FILE
    Y_TOPIC_FILE = s.Y_TOPIC_FILE

    X_COORDINATE_EVALUATION_FILE = s.X_COORDINATE_EVALUATION_FILE
    Y_COORDINATE_EVALUATION_FILE = s.Y_COORDINATE_EVALUATION_FILE
    X_MODE_EVALUATION_FILE = s.X_MODE_EVALUATION_FILE
    X_GRID_EVALUATION_FILE = s.X_GRID_EVALUATION_FILE
    Y_GRID_EVALUATION_FILE = s.Y_GRID_EVALUATION_FILE
    Y_MODE_EVALUATION_FILE = s.Y_MODE_EVALUATION_FILE
    X_TOPIC_EVALUATION_FILE = s.X_TOPIC_EVALUATION_FILE
    Y_TOPIC_EVALUATION_FILE = s.Y_TOPIC_EVALUATION_FILE

    X_COORDINATE_EVALUATION_SMALL_FILE = s.X_COORDINATE_EVALUATION_SMALL_FILE
    Y_COORDINATE_EVALUATION_SMALL_FILE = s.Y_COORDINATE_EVALUATION_SMALL_FILE
    X_GRID_EVALUATION_SMALL_FILE = s.X_GRID_EVALUATION_SMALL_FILE
    Y_GRID_EVALUATION_SMALL_FILE = s.Y_GRID_EVALUATION_SMALL_FILE
    X_MODE_EVALUATION_SMALL_FILE = s.X_MODE_EVALUATION_SMALL_FILE
    Y_MODE_EVALUATION_SMALL_FILE = s.Y_MODE_EVALUATION_SMALL_FILE
    X_TOPIC_EVALUATION_SMALL_FILE = s.X_TOPIC_EVALUATION_SMALL_FILE
    Y_TOPIC_EVALUATION_SMALL_FILE = s.Y_TOPIC_EVALUATION_SMALL_FILE

    TEMPORAL_DATAFRAME = s.TEMPORAL_DATAFRAME
    LSI_TOPIC_FILE = s.LSI_TOPIC_FILE
    LDA_TOPIC_FILE = s.LDA_TOPIC_FILE
    DOC2VEC_TOPIC_FILE = s.DOC2VEC_TOPIC_FILE

    cores = mp.cpu_count()
    logger.info("Using %s cores" % cores)
    pool = mp.Pool(cores)


    if EXPERIMENT_ENVIRONMENT == "remote":
        engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet_remote()
    elif EXPERIMENT_ENVIRONMENT == "local":
        engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet_ssh()

    temporal_index = utility_spatiotemporal_index.define_temporal_index(EXPERIMENT_PARAMETERS)
    spatial_index = utility_spatiotemporal_index.define_spatial_index(EXPERIMENT_PARAMETERS)
    topic_array = np.load(LDA_TOPIC_FILE) # Change for changing topic models

    # Training data
    df_users_regularized = load_gps_trajectory_db_to_dataframe_parallel(GPS_TABLE_RAW_NAME, GPS_TABLE_REGULARIZED_NAME, PROFILE_TABLE_NAME, EXPERIMENT_PARAMETERS, conn, temporal_index, topic_array)
    df_users_regularized.to_hdf(TEMPORAL_DATAFRAME, "df_users_regularized")

    # df_users_regularized = pd.read_hdf(TEMPORAL_DATAFRAME, "df_users_regularized")
    X_coordinate_all, y_coordinate_all, X_grid_all, y_grid_all, X_mode_all, y_mode_all, X_topic_all, y_topic_all = convert_dataframe_to_coordinate_mode_and_grid_mode_topic_sequences(df_users_regularized, EXPERIMENT_PARAMETERS)

    # logger.info(X_coordinate_all[0])
    # logger.info(y_coordinate_all[0])
    # logger.info(X_grid_all[0])
    # logger.info(y_grid_all[0])
    # logger.info(X_mode_all[0])
    # logger.info(y_mode_all[0])
    # logger.info(X_topic_all[0])
    # logger.info(y_topic_all[0])

    np.save(file=X_COORDINATE_FILE, arr=X_coordinate_all)
    np.save(file=Y_COORDINATE_FILE, arr=y_coordinate_all)
    np.save(file=X_GRID_FILE, arr=X_grid_all)
    np.save(file=Y_GRID_FILE, arr=y_grid_all)
    np.save(file=X_MODE_FILE, arr=X_mode_all)
    np.save(file=Y_MODE_FILE, arr=y_mode_all)
    np.save(file=X_TOPIC_FILE, arr=X_topic_all)
    np.save(file=Y_TOPIC_FILE, arr=y_topic_all)

    # Test data

    df_users_regularized_evaluation = load_gps_trajectory_db_to_dataframe_parallel_evaluation(GPS_TABLE_RAW_NAME, GPS_TABLE_REGULARIZED_NAME, PROFILE_TABLE_NAME, EXPERIMENT_PARAMETERS, conn, temporal_index, topic_array)
    df_users_regularized_evaluation.to_hdf(TEMPORAL_DATAFRAME, "df_users_regularized_evaluation")
    # df_users_regularized_evaluation = pd.read_hdf(TEMPORAL_DATAFRAME, "df_users_regularized_evaluation")
    X_coordinate_all_evaluation, y_coordinate_all_evaluation, X_grid_all_evaluation, y_grid_all_evaluation, X_mode_all_evaluation, y_mode_all_evaluation, X_topic_all_evaluation, y_topic_all_evaluation = convert_dataframe_to_coordinate_mode_and_grid_mode_topic_sequences(df_users_regularized_evaluation, EXPERIMENT_PARAMETERS)

    np.save(file=X_COORDINATE_EVALUATION_FILE, arr=X_coordinate_all_evaluation)
    np.save(file=Y_COORDINATE_EVALUATION_FILE, arr=y_coordinate_all_evaluation)
    np.save(file=X_GRID_EVALUATION_FILE, arr=X_grid_all_evaluation)
    np.save(file=Y_GRID_EVALUATION_FILE, arr=y_grid_all_evaluation)
    np.save(file=X_MODE_EVALUATION_FILE, arr=X_mode_all_evaluation)
    np.save(file=Y_MODE_EVALUATION_FILE, arr=y_mode_all_evaluation)
    np.save(file=X_TOPIC_EVALUATION_FILE, arr=X_topic_all_evaluation)
    np.save(file=Y_TOPIC_EVALUATION_FILE, arr=y_topic_all_evaluation)

    # Test small data
    df_users_regularized_evaluation = load_gps_trajectory_db_to_dataframe_parallel_evaluation(GPS_TABLE_RAW_NAME, GPS_TABLE_REGULARIZED_NAME, PROFILE_TABLE_NAME_SMALL_TARGET, EXPERIMENT_PARAMETERS, conn, temporal_index, topic_array)
    df_users_regularized_evaluation.to_hdf(TEMPORAL_DATAFRAME, "df_users_regularized_evaluation")
    # df_users_regularized_evaluation = pd.read_hdf(TEMPORAL_DATAFRAME, "df_users_regularized_evaluation")
    X_coordinate_all_evaluation, y_coordinate_all_evaluation, X_grid_all_evaluation, y_grid_all_evaluation, X_mode_all_evaluation, y_mode_all_evaluation, X_topic_all_evaluation, y_topic_all_evaluation = convert_dataframe_to_coordinate_mode_and_grid_mode_topic_sequences(df_users_regularized_evaluation, EXPERIMENT_PARAMETERS)

    np.save(file=X_COORDINATE_EVALUATION_SMALL_FILE, arr=X_coordinate_all_evaluation)
    np.save(file=Y_COORDINATE_EVALUATION_SMALL_FILE, arr=y_coordinate_all_evaluation)
    np.save(file=X_GRID_EVALUATION_SMALL_FILE, arr=X_grid_all_evaluation)
    np.save(file=Y_GRID_EVALUATION_SMALL_FILE, arr=y_grid_all_evaluation)
    np.save(file=X_MODE_EVALUATION_SMALL_FILE, arr=X_mode_all_evaluation)
    np.save(file=Y_MODE_EVALUATION_SMALL_FILE, arr=y_mode_all_evaluation)
    np.save(file=X_TOPIC_EVALUATION_SMALL_FILE, arr=X_topic_all_evaluation)
    np.save(file=Y_TOPIC_EVALUATION_SMALL_FILE, arr=y_topic_all_evaluation)

    elapsed = str((datetime.now() - start_time))
    logger.info(f"Finished in: {elapsed}")
    atexit.register(s.exit_handler, __file__, elapsed)

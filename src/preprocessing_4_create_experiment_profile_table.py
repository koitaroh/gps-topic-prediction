import requests
import json
import csv
import multiprocessing as mp
from itertools import repeat
import os
import atexit
import time
import random
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sqlalchemy
import pickle
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import geopandas as gpd
import swifter
from datetime import datetime, timedelta, time, date
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, LabelEncoder
from tqdm import tqdm
import jpgrid
from sqlalchemy import create_engine
from sqlalchemy.sql import text


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
import jpgrid
np.random.seed(1)
random.seed(1)

# def nearest(items, pivot):
#     return min(items, key=lambda x: abs(x - pivot))


def create_profile_table(EXPERIMENT_PARAMETERS, PROFILE_TABLE_NAME, conn):
    sql_this = f"""
        CREATE table {PROFILE_TABLE_NAME} as
        SELECT uid
        FROM gps_interpolated
        where (timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}')
        and (uid in (
            select uid from gps_interpolated 
            GROUP BY uid
            having (min(timestamp_from) < '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}') and (max(timestamp_from) > '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}')
        ))
        and (uid in (
            select uid from gps_interpolated 
            where timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END']}'
            GROUP BY uid
        ))
        and (uid in (
            select uid from gps_interpolated 
            where timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}'
            GROUP BY uid
        ))
        GROUP BY uid
        having (min(latitude) >= {EXPERIMENT_PARAMETERS['AOI'][1]}) and (max(latitude) <= {EXPERIMENT_PARAMETERS['AOI'][3]})
        and (min(longitude) >= {EXPERIMENT_PARAMETERS['AOI'][0]}) and (max(longitude) <= {EXPERIMENT_PARAMETERS['AOI'][2]})
        limit {EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE']}
    """
    try:
        conn.execute(text(sql_this))
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")

    sql = f"""
        ALTER TABLE {PROFILE_TABLE_NAME}
        ADD obs_s_lon double precision,
        ADD obs_s_lat double precision,
        ADD obs_e_lon double precision,
        ADD obs_e_lat double precision,
        ADD pred_s_lon double precision,
        ADD pred_s_lat double precision,
        ADD pred_e_lon double precision,
        ADD pred_e_lat double precision,
        ADD obs_s_jpgrid text,
        ADD obs_e_jpgrid text,
        ADD pred_s_jpgrid text,
        ADD pred_e_jpgrid text,
        ADD preprocessing integer
    """
    conn.execute(text(sql))
    conn.commit()
    return None

def update_profile_table(EXPERIMENT_PARAMETERS, PROFILE_TABLE_NAME, conn, metadata, engine):
    sql = f"""
        select uid from {PROFILE_TABLE_NAME} where obs_s_lon is NULL
    """
    uid_df = pd.read_sql_query(text(sql), conn)
    for index, row in tqdm(uid_df.iterrows()):
        uid = int(row['uid'])
        sql = f"""
            SELECT uid, timestamp_from, latitude, longitude
            FROM gps_interpolated
            where uid = {uid}
            and (latitude between {EXPERIMENT_PARAMETERS['AOI'][1]} and {EXPERIMENT_PARAMETERS['AOI'][3]})
            and (longitude between {EXPERIMENT_PARAMETERS['AOI'][0]} and {EXPERIMENT_PARAMETERS['AOI'][2]})
            and timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}'
        """

        dt_observation_start = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START'])
        dt_observation_end = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END'])
        dt_prediction_start = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START'])
        dt_prediction_end = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END'])

        gps_gdf = pd.read_sql_query(text(sql), conn, parse_dates=['timestamp_from'])
        # # gps_gdf = gpd.GeoDataFrame(gps_gdf, geometry=gpd.points_from_xy(gps_gdf.longitude, gps_gdf.latitude)) # For geopandas 0.5.0 or later
        gps_gdf = gpd.GeoDataFrame(gps_gdf, geometry=[Point(x, y) for x, y in zip(gps_gdf.longitude, gps_gdf.latitude)]) # For geopandas 0.4.0
        gps_gdf = gps_gdf.drop_duplicates(subset='timestamp_from', keep='first')
        gps_gdf.set_index('timestamp_from', inplace=True)
        gps_gdf = gps_gdf.sort_index()
        point_observation_start = gps_gdf.iloc[gps_gdf.index.get_loc(dt_observation_start, method='backfill')]
        point_observation_end = gps_gdf.iloc[gps_gdf.index.get_loc(dt_observation_end, method='pad')]
        point_prediction_start = gps_gdf.iloc[gps_gdf.index.get_loc(dt_prediction_start, method='backfill')]
        point_prediction_end = gps_gdf.iloc[gps_gdf.index.get_loc(dt_prediction_end, method='pad')]

    
        table_profile = sqlalchemy.Table(PROFILE_TABLE_NAME, metadata, autoload_with=engine, schema='public')
        query = table_profile.update()\
            .values(
            obs_s_lon=point_observation_start['longitude'],
            obs_s_lat=point_observation_start['latitude'],
            obs_e_lon=point_observation_end['longitude'],
            obs_e_lat=point_observation_end['latitude'],
            pred_s_lon=point_prediction_start['longitude'],
            pred_s_lat=point_prediction_start['latitude'],
            pred_e_lon=point_prediction_end['longitude'],
            pred_e_lat=point_prediction_end['latitude'],
            obs_s_jpgrid=jpgrid.encodeBase(point_observation_start['latitude'],point_observation_start['longitude']),
            obs_e_jpgrid=jpgrid.encodeBase(point_observation_end['latitude'],point_observation_end['longitude']),
            pred_s_jpgrid=jpgrid.encodeBase(point_prediction_start['latitude'], point_prediction_start['longitude']),
            pred_e_jpgrid=jpgrid.encodeBase(point_prediction_end['latitude'], point_prediction_end['longitude']))\
            .where(table_profile.c.uid == uid)
        conn.execute(query)
        conn.commit()
    return None

def create_profile_small_table(EXPERIMENT_PARAMETERS, PROFILE_TABLE_NAME, conn):
    sql = f"""
        CREATE table {PROFILE_TABLE_NAME} as
        SELECT uid
        FROM gps_interpolated
        WHERE uid IN (
            SELECT DISTINCT uid
            FROM gps_interpolated
            WHERE 
                timestamp_from BETWEEN '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}'
                AND (latitude BETWEEN {EXPERIMENT_PARAMETERS['AOI_SMALL'][1]} AND {EXPERIMENT_PARAMETERS['AOI_SMALL'][3]})
                AND (longitude BETWEEN {EXPERIMENT_PARAMETERS['AOI_SMALL'][0]} AND {EXPERIMENT_PARAMETERS['AOI_SMALL'][2]})
        )
        and (timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}')
        and (uid in (
            select uid from gps_interpolated 
            GROUP BY uid
            having (min(timestamp_from) < '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}') and (max(timestamp_from) > '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}')
        ))
        and (uid in (
            select uid from gps_interpolated 
            where timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END']}'
            GROUP BY uid
        ))
        and (uid in (
            select uid from gps_interpolated 
            where timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}'
            GROUP BY uid
        ))
        GROUP BY uid
        having (min(latitude) >= {EXPERIMENT_PARAMETERS['AOI'][1]}) and (max(latitude) <= {EXPERIMENT_PARAMETERS['AOI'][3]})
        and (min(longitude) >= {EXPERIMENT_PARAMETERS['AOI'][0]}) and (max(longitude) <= {EXPERIMENT_PARAMETERS['AOI'][2]})
        limit {EXPERIMENT_PARAMETERS['SAMPLE_USER_SIZE']}
    """
    conn.execute(text(sql))
    conn.commit()

    sql = f"""
        ALTER TABLE {PROFILE_TABLE_NAME}
        ADD obs_s_lon double precision,
        ADD obs_s_lat double precision,
        ADD obs_e_lon double precision,
        ADD obs_e_lat double precision,
        ADD pred_s_lon double precision,
        ADD pred_s_lat double precision,
        ADD pred_e_lon double precision,
        ADD pred_e_lat double precision,
        ADD obs_s_jpgrid text,
        ADD obs_e_jpgrid text,
        ADD pred_s_jpgrid text,
        ADD pred_e_jpgrid text,
        ADD preprocessing integer
    """
    conn.execute(text(sql))
    conn.commit()
    return None


if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    EXPERIMENT_ENVIRONMENT = s.EXPERIMENT_ENVIRONMENT
    SCENARIO = s.SCENARIO
    PROFILE_TABLE_NAME = s.PROFILE_TABLE_NAME
    PROFILE_TABLE_NAME_SMALL_TARGET = s.PROFILE_TABLE_NAME_SMALL_TARGET

    engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet(EXPERIMENT_ENVIRONMENT)
    create_profile_table(EXPERIMENT_PARAMETERS, PROFILE_TABLE_NAME, conn)
    update_profile_table(EXPERIMENT_PARAMETERS, PROFILE_TABLE_NAME, conn, metadata, engine)
    create_profile_small_table(EXPERIMENT_PARAMETERS, PROFILE_TABLE_NAME_SMALL_TARGET, conn)
    update_profile_table(EXPERIMENT_PARAMETERS, PROFILE_TABLE_NAME_SMALL_TARGET, conn, metadata, engine)


    elapsed = str((datetime.now() - start_time))
    logger.info(f"Finished in: {elapsed}")
    atexit.register(s.exit_handler, __file__, elapsed)

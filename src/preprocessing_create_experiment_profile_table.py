import requests
import json
import csv
import multiprocessing as mp
from itertools import repeat
import os
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
from datetime import datetime, timedelta, time, date
from geopy.distance import vincenty, great_circle
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, LabelEncoder
from tqdm import tqdm
import jpgrid

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
import utility_database
import jpgrid
np.random.seed(1)
random.seed(1)

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def create_profile_table(EXPERIMENT_PARAMETERS, SCENARIO, conn):
    sql = f"""
        CREATE table gps_topic_prediction_profile_{SCENARIO} as
        SELECT uid
        FROM gps_interpolated
        where (latitude between {EXPERIMENT_PARAMETERS['AOI'][1]} and {EXPERIMENT_PARAMETERS['AOI'][3]})
        and (longitude between {EXPERIMENT_PARAMETERS['AOI'][0]} and {EXPERIMENT_PARAMETERS['AOI'][2]})
        and timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}'
        GROUP BY uid
    """
    conn.execute(sql)

    sql = f"""
        ALTER TABLE gps_topic_prediction_profile_{SCENARIO}
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
    conn.execute(sql)
    return None


def update_profile_table(EXPERIMENT_PARAMETERS, SCENARIO, conn):
    sql = f"""
        select uid from gps_topic_prediction_profile_{SCENARIO} where obs_s_lon is NULL
    """
    # limit 1000 for prototyping
    uid_df = pd.read_sql_query(sql, conn)
    for index, row in tqdm(uid_df.iterrows()):
        # print(row['uid'])
        uid = int(row['uid'])
        sql = f"""
            SELECT uid, timestamp_from, latitude, longitude
            FROM gps_interpolated
            where uid = {uid}
            and (latitude between {EXPERIMENT_PARAMETERS['AOI'][1]} and {EXPERIMENT_PARAMETERS['AOI'][3]})
            and (longitude between {EXPERIMENT_PARAMETERS['AOI'][0]} and {EXPERIMENT_PARAMETERS['AOI'][2]})
            and timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}'
        """
        gps_gdf = pd.read_sql_query(sql, conn, parse_dates=['timestamp_from'])
        # # gps_gdf = gpd.GeoDataFrame(gps_gdf, geometry=gpd.points_from_xy(gps_gdf.longitude, gps_gdf.latitude)) # For geopandas 0.5.0 or later
        gps_gdf = gpd.GeoDataFrame(gps_gdf, geometry=[Point(x, y) for x, y in zip(gps_gdf.longitude, gps_gdf.latitude)]) # For geopandas 0.4.0
        gps_gdf = gps_gdf.drop_duplicates(subset='timestamp_from', keep='first')
        gps_gdf.set_index('timestamp_from', inplace=True)
        gps_gdf = gps_gdf.sort_index()
        point_observation_start = gps_gdf.iloc[gps_gdf.index.get_loc(dt_observation_start, method='nearest')]
        point_observation_end = gps_gdf.iloc[gps_gdf.index.get_loc(dt_observation_end, method='nearest')]
        point_prediction_start = gps_gdf.iloc[gps_gdf.index.get_loc(dt_prediction_start, method='nearest')]
        point_prediction_end = gps_gdf.iloc[gps_gdf.index.get_loc(dt_prediction_end, method='nearest')]
        table_profile = sqlalchemy.Table(f"gps_topic_prediction_profile_{SCENARIO}", metadata, autoload=True, autoload_with=engine)
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
    return None


if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    EXPERIMENT_ENVIRONMENT = s.EXPERIMENT_ENVIRONMENT
    SCENARIO = s.SCENARIO


    dt_observation_start = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START'])
    dt_observation_end = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END'])
    dt_prediction_start = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START'])
    dt_prediction_end = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END'])

    slack_client = s.sc

    if EXPERIMENT_ENVIRONMENT == "remote":
        engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet_remote()
    elif EXPERIMENT_ENVIRONMENT == "local":
        engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet_ssh()

    create_profile_table(EXPERIMENT_PARAMETERS, SCENARIO, conn)
    update_profile_table(EXPERIMENT_PARAMETERS, SCENARIO, conn)


    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__ + " is finished.")
    logger.info("Done.")
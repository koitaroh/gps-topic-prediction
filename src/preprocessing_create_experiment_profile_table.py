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
import folium
import sqlalchemy
# import tables
import pickle
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import shapely
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


if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    dt_observation_start = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START'])
    dt_observation_end = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_END'])
    dt_prediction_start = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_START'])
    dt_prediction_end = pd.to_datetime(EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END'])

    slack_client = s.sc

    engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet_ssh()
    engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet_ssh()

    profile_list = []

    sql = f"""
        SELECT uid, timestamp_from, latitude, longitude
        FROM gps_interpolated
        where (latitude between {EXPERIMENT_PARAMETERS['AOI'][1]} and {EXPERIMENT_PARAMETERS['AOI'][3]}) 
        and (longitude between {EXPERIMENT_PARAMETERS['AOI'][0]} and {EXPERIMENT_PARAMETERS['AOI'][2]})
        and timestamp_from between '{EXPERIMENT_PARAMETERS['TRAINING_OBSERVATION_START']}' and '{EXPERIMENT_PARAMETERS['TRAINING_PREDICTION_END']}'
        limit 10000000
    """

    gps_gdf = pd.read_sql_query(sql, conn, parse_dates=['timestamp_from'])
    gps_gdf = gpd.GeoDataFrame(gps_gdf, geometry=gpd.points_from_xy(gps_gdf.longitude, gps_gdf.latitude))

    gps_gdf_group = gps_gdf.groupby("uid", as_index=False)
    for name, group in tqdm(gps_gdf_group):
        group = group.drop_duplicates(subset='timestamp_from', keep='first')
        group.set_index('timestamp_from', inplace=True)
        group = group.sort_index()
        point_observation_start = group.iloc[group.index.get_loc(dt_observation_start, method='nearest')]
        point_observation_end = group.iloc[group.index.get_loc(dt_observation_end, method='nearest')]
        point_prediction_start = group.iloc[group.index.get_loc(dt_prediction_start, method='nearest')]
        point_prediction_end = group.iloc[group.index.get_loc(dt_prediction_end, method='nearest')]
        profile_list.append({'uid': name,
                             'obs_s_lon': point_observation_start['longitude'],
                             'obs_s_lat': point_observation_start['latitude'],
                             'obs_e_lon': point_observation_end['longitude'],
                             'obs_e_lat': point_observation_end['latitude'],
                             'pred_s_lon': point_prediction_start['longitude'],
                             'pred_s_lat': point_prediction_start['latitude'],
                             'pred_e_lon': point_prediction_end['longitude'],
                             'pred_e_lat': point_prediction_end['latitude'],
                             'obs_s_jpgrid': jpgrid.encodeBase(point_observation_start['latitude'],
                                                               point_observation_start['longitude']),
                             'obs_e_jpgrid': jpgrid.encodeBase(point_observation_end['latitude'],
                                                               point_observation_end['longitude']),
                             'pred_s_jpgrid': jpgrid.encodeBase(point_prediction_start['latitude'],
                                                                point_prediction_start['longitude']),
                             'pred_e_jpgrid': jpgrid.encodeBase(point_prediction_end['latitude'],
                                                                point_prediction_end['longitude'])})
    profile_df = pd.DataFrame(profile_list)
    # print(profile_df)
    profile_df.to_sql(EXPERIMENT_PARAMETERS['EXPERIMENT_NAME'], engine, if_exists='replace')

    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__ + " is finished.")
    logger.info("Done.")
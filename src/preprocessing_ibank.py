import requests
import json
import csv
import os
import time
import sqlalchemy
import pandas
import pickle
import numpy
import datetime
from datetime import datetime, timedelta
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy

import settings as s
import utility_database

# Logging ver. 2016-07-12
from logging import handlers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('log.log', maxBytes=1000000, backupCount=3)  # file handler
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # console handler
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Initializing %s', __name__)


def load_db_to_dataframe(TABLE_NAME, engine, conn, metadata):
    logger.info("Loading DB table %s to dataframe" % TABLE_NAME)
    sql = "select user_uri as uid, lat as y, lng as x, time as timestamp, purpose from gps_log_2 where purpose is not null"
    df_db= pandas.read_sql(sql=sql, con=conn, parse_dates=['timestamp'])
    df_db = df_db.sort_values(by=['uid', 'timestamp'])
    # print(df_db)
    return df_db


if __name__ == '__main__':
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    TEST_DB_TABLE_NAME = s.TEST_DB_TABLE_NAME
    TRANSFER_CSV_FILTERED = s.TRANSFER_CSV_FILTERED

    engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet()
    db_df = load_db_to_dataframe(TEST_DB_TABLE_NAME, engine, conn, metadata)

    df_user_filtered = filter_gps.filter_users_with_experiment_setting(db_df, EXPERIMENT_PARAMETERS)
    temporal_indices = filter_gps.define_temporal_indices(EXPERIMENT_PARAMETERS)
    # print(temporal_indices)
    #
    df_users_regularized = filter_gps.apply_slice_to_users(df_user_filtered, temporal_indices)
    filter_gps.save_dataframe_to_csv(df_users_regularized, TRANSFER_CSV_FILTERED, regularized=True)
    # # print(df_user_filtered)
    # # csv_to_list(TEST_CSV)


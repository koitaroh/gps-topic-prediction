import requests
import json
import csv
import multiprocessing as mp
from itertools import repeat
import os
import sys
import time
import random
import sqlalchemy
import pandas
import tables
import pickle
import numpy
import numpy as np
from datetime import datetime, timedelta
import geopy
from geopy.distance import vincenty
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import smart_open
import random
import multiprocessing
import pymysql
import configparser
from tqdm import tqdm
import geojson
from geojson import LineString, FeatureCollection, Feature
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError

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


def upload_dir_to_dropbox(dir, path_in_dropbox):
    ACCESS_TOKEN = "vzhi0bre3NIAAAAAAADXopLD634TYIaIXdW5DA7DG0vn2c-iKp1nRI_8OL7FfXn1"
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    dbx.users_get_current_account()
    for root, dirs, files in os.walk(dir):
        for fn in files:
            if fn[0] != '.':
                file = root + fn
                with open(file, 'rb') as f:
                    # We use WriteMode=overwrite to make sure that the settings in the file
                    # are changed on upload
                    print("Uploading " + file + " to Dropbox as " + path_in_dropbox+fn + "...")
                    try:
                        dbx.files_upload(f.read(), path_in_dropbox+fn, mode=WriteMode('overwrite'))
                    except ApiError as err:
                        # This checks for the specific error where a user doesn't have
                        # enough Dropbox space quota to upload this file
                        if (err.error.is_path() and
                                err.error.get_path().reason.is_insufficient_space()):
                            sys.exit("ERROR: Cannot back up; insufficient space.")
                        elif err.user_message_text:
                            print(err.user_message_text)
                            sys.exit()
                        else:
                            print(err)
                            sys.exit()
    return None


if __name__ == '__main__':
    TEST_DIR = "/Users/koitaroh/Downloads/training/"
    TEST_DROPBOX_PATH = "/training/"

    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR
    TRAINING_DIR = s.TRAINING_DIR
    EVALUATION_DIR = s.EVALUATION_DIR

    upload_dir_to_dropbox(FIGURE_DIR, "/" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + "/figures/")
    upload_dir_to_dropbox(TRAINING_DIR, "/" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + "/training/")
    upload_dir_to_dropbox(EVALUATION_DIR, "/" + EXPERIMENT_PARAMETERS["EXPERIMENT_NAME"] + "/evaluation/")
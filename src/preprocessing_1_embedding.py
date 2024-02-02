import multiprocessing as mp
from itertools import repeat
import os
import atexit
import random
import pandas as pd
from collections import OrderedDict
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import geopy
from geopy.distance import geodesic



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
import utility_spatiotemporal_index
import jpgrid
np.random.seed(1)
random.seed(1)


def create_labelencoder(MESHCODE_FILE, LE_GRID_CLASSES_FILE):
    le_grid = LabelEncoder()
    df = pd.read_csv(MESHCODE_FILE, delimiter=',', header=None, dtype='str')
    mesh_list = df[0].values.tolist()
    le_grid.fit(mesh_list)
    logger.info("Standard mesh: %s", le_grid.classes_)
    logger.info("Size of spatial index: %s", len(le_grid.classes_))
    np.save(LE_GRID_CLASSES_FILE, le_grid.classes_)
    return None


def create_distance_matrix_from_meshcode_2nd(MESHCODE_FILE, CITYEMD_DISTANCE_MATRIX_FILE):
    df = pd.read_csv(MESHCODE_FILE, delimiter=',', header=None, dtype='str')
    mesh_list = df[0].values.tolist()
    # print(len(mesh_list))
    distance_matrix = np.zeros((len(mesh_list), len(mesh_list)))
    # print(distance_matrix)
    # print(distance_matrix.shape)
    for i, meshcode_i in enumerate(mesh_list):
        y_i, x_i = jpgrid.decode(meshcode_i)
        # print(jpgrid.decode(meshcode_i))
        for j, meshcode_j in enumerate(mesh_list):
            y_j, x_j = jpgrid.decode(meshcode_j)
            distance = geodesic((y_i, x_i), (y_j, x_j)).kilometers
            distance_matrix[i, j] = distance
            # print(distance)
    # print(distance_matrix)
    np.save(CITYEMD_DISTANCE_MATRIX_FILE, distance_matrix)
    return None


def create_mode_labelbinaryencoder(MODE_LIST, LB_MODE_CLASSES_FILE):
    lb_mode = LabelBinarizer()
    lb_mode.fit(MODE_LIST)
    logger.info("Mode classes: %s", lb_mode.classes_)
    np.save(LB_MODE_CLASSES_FILE, lb_mode.classes_)
    return None


# def test_preprocessing_embedding(self):
#     # WIP. Leaving sample code
#     expected_labels = ['I-ORG', 'O', 'I-PER', 'O', 'O', 'I-LOC', 'O']
#
#     fields = instances[0].fields
#     tokens = [t.text for t in fields['tokens'].tokens]
#     assert tokens == ['U.N.', 'official', 'Ekeus', 'heads', 'for', 'Baghdad', '.']
#     assert fields["tags"].labels == expected_labels
#
#     fields = instances[1].fields
#     tokens = [t.text for t in fields['tokens'].tokens]
#     assert tokens == ['AI2', 'engineer', 'Joel', 'lives', 'in', 'Seattle', '.']
#     assert fields["tags"].labels == expected_labels


if __name__ == '__main__':
    # slack_client = s.sc
    MESHCODE_FILE = s.MESHCODE_FILE
    MESHCODE_FILE_2ND = s.MESHCODE_FILE_2ND
    CITYEMD_DISTANCE_MATRIX_FILE = s.CITYEMD_DISTANCE_MATRIX_FILE
    LE_GRID_CLASSES_FILE = s.LE_GRID_CLASSES_FILE
    LB_MODE_CLASSES_FILE = s.LB_MODE_CLASSES_FILE
    MODE_LIST = s.MODE_LIST

    create_labelencoder(MESHCODE_FILE, LE_GRID_CLASSES_FILE)
    create_mode_labelbinaryencoder(MODE_LIST, LB_MODE_CLASSES_FILE)

    create_distance_matrix_from_meshcode_2nd(MESHCODE_FILE_2ND, CITYEMD_DISTANCE_MATRIX_FILE)

    elapsed = str((datetime.now() - start_time))
    logger.info(f"Finished in: {elapsed}")
    atexit.register(s.exit_handler, __file__, elapsed)  # Notification for exit

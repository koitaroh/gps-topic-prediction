import requests
import json
import csv
import os
import time
import random
import math
import sqlalchemy
import geojson
from geojson import LineString, FeatureCollection, Feature
import pandas
import pickle
import numpy as np
import tensorflow as tf
import h5py
from geopy.distance import vincenty
from datetime import datetime, timedelta
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed, LSTM, SimpleRNN, GRU, Input, Flatten, Bidirectional
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

import settings as s
import load_dataset
import preprocessing_gps

np.random.seed(7)
random.seed(7)


def load_grid_dataset(X_GRID_FILE, Y_GRID_FILE, EXPERIMENT_PARAMETERS):
    X_all = np.load(X_GRID_FILE)
    y_all = np.load(Y_GRID_FILE)

    X_all_shape = X_all.shape
    y_all_shape = y_all.shape

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, test_size=0.2, random_state=7)
    X_train_all, y_train_all = load_dataset.create_full_training_sample(X_train, y_train)
    return X_train_all, y_train_all, X_test, y_test


def convert_spatial_index_to_raw_coordinates(EXPERIMENT_PARAMETERS, spatial_index, spatial_index_number):
    spatial_index_number = int(spatial_index_number)
    y_index = spatial_index_number // spatial_index[3]
    x_index = spatial_index_number - (y_index * spatial_index[3])
    print(x_index)
    print(y_index)
    x = EXPERIMENT_PARAMETERS["AOI"][0] + (spatial_index[0] * x_index) + (spatial_index[0] / 2)
    y = EXPERIMENT_PARAMETERS["AOI"][1] + (spatial_index[1] * y_index) + (spatial_index[1] / 2)
    return x, y


def convert_spatial_index_to_latitude(spatial_index_number, EXPERIMENT_PARAMETERS, spatial_index):
    y_index = spatial_index_number // spatial_index[3]
    y = EXPERIMENT_PARAMETERS["AOI"][1] + (spatial_index[1] * y_index) + (spatial_index[1] / 2)
    return y


def convert_spatial_index_to_longitude(spatial_index_number, EXPERIMENT_PARAMETERS, spatial_index):
    y_index = spatial_index_number // spatial_index[3]
    x_index = spatial_index_number - (y_index * spatial_index[3])
    x = EXPERIMENT_PARAMETERS["AOI"][0] + (spatial_index[0] * x_index) + (spatial_index[0] / 2)
    return x


def convert_spatial_index_array_to_coordinate_array(input_array, EXPERIMENT_PARAMETERS, spatial_index, Multistep=False):
    if Multistep:
        X_shape = input_array.shape
        # print(X_shape) # (18, 12, 1)


        x_array = np.apply_along_axis(convert_spatial_index_to_longitude, 0, input_array, EXPERIMENT_PARAMETERS, spatial_index)
        y_array = np.apply_along_axis(convert_spatial_index_to_latitude, 0, input_array, EXPERIMENT_PARAMETERS, spatial_index)
        latlon_array = np.concatenate((y_array, x_array), axis=2)

    else:
        x_array = np.apply_along_axis(convert_spatial_index_to_longitude, 0, input_array, EXPERIMENT_PARAMETERS,
                                      spatial_index)
        y_array = np.apply_along_axis(convert_spatial_index_to_latitude, 0, input_array, EXPERIMENT_PARAMETERS,
                                      spatial_index)
        latlon_array = np.concatenate((y_array, x_array), axis=1)
    return latlon_array


def training_gb(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, max_s_index):

    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    y_train_shape = y_train.shape
    y_test_shape = y_test.shape
    X_train = X_train.reshape(X_train_shape[0], X_train_shape[1])
    X_test = X_test.reshape(X_test_shape[0], X_test_shape[1])
    print(X_train)
    print(X_train.shape)
    print(y_train)
    print(y_train.shape)

    lb = LabelBinarizer()
    X_train_flat = X_train.flatten()
    X_test_flat = X_test.flatten()
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()
    X_y_flat = np.concatenate((X_train_flat, X_test_flat, y_train_flat, y_test_flat), axis=0)
    lb.fit(X_y_flat)
    print(lb.classes_)
    X_train_vector = lb.transform(X_train_flat)
    X_train_vector = X_train_vector.reshape(X_train_shape[0], X_train_shape[1], -1)
    X_test_vector = lb.transform(X_test_flat)
    X_test_vector = X_test_vector.reshape(X_test_shape[0], X_test_shape[1], -1)
    y_train_vector = lb.transform(y_train_flat)
    y_train_vector = y_train_vector.reshape(y_train_shape[0], -1)
    y_test_vector = lb.transform(y_test_flat)
    y_test_vector = y_test_vector.reshape(y_test_shape[0], -1)
    print(X_train_vector)
    print(X_train_vector.shape)
    print(y_train_vector)
    print(y_train_vector.shape)

    gb = GradientBoostingClassifier()
    return None



def training_rnn_grid(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, max_s_index, MODEL_FILE, MODEL_WEIGHT_FILE_LSTM_GRID, FIGURE_DIR):
    '''
    :param X_train: Training data with shape
    :param y_train: The number of feature maps we'd like to calculate
    :param X_test: The filter width
    :param y_test: The stride
    :param EXPERIMENT_PARAMETERS: Experiment parameters
    :return: none
    '''

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    y_test_one_step = y_test[:, 0, :]
    print('y_test shape:', y_test_one_step.shape)
    input_shape = X_train.shape[1:]
    print('input shape:', input_shape)

    hidden_neurons = 128
    batch_size = 50
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    x_input = Input(shape=input_shape)
    flatten = Flatten()(x_input)
    embedding = Embedding(input_dim=max_s_index, output_dim=256, input_length=X_train.shape[1])(flatten)

    # uni-directional
    rnn_1 = SimpleRNN(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-1')(embedding)
    rnn_2 = SimpleRNN(hidden_neurons, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='lstm-2')(rnn_1)
    main_output = Dense(max_s_index, name='dense-1', activation='softmax')(rnn_2)

    # bi-directional
    # bi_lstm = Bidirectional(LSTM(hidden_neurons, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='bi-lstm'))(embedding)
    # main_output = Dense(max_s_index, name='dense-1', activation='softmax')(bi_lstm)

    model = Model(inputs=x_input, outputs=main_output)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])
    model.summary()


    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=50, validation_data=(X_test, y_test_one_step), callbacks=[es_cb])

    model.save(MODEL_FILE)
    model.save_weights(MODEL_WEIGHT_FILE_LSTM_GRID)

    # print(model.metrics_names)
    loss, accuracy, sparse_top_k_accuracy = model.evaluate(X_test, y_test_one_step, batch_size=300)
    logger.info('sparse_categorical_crossentropy score: %s' % loss)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss (sparse_categorical_crossentropy)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(FIGURE_DIR + "accuracy_sparse_categorical_crossentropy.png")
    plt.close()

    logger.info("sparse_categorical_accuracy: %s" % accuracy)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy (sparse_categorical_accuracy)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(FIGURE_DIR + "accuracy_sparse_categorical_accuracy.png")
    plt.close()

    return model


def prediction_multiple_steps(model, X, y, spatial_index, EXPERIMENT_PARAMETERS):
    PREDICTION_OUTPUT_LENGTH = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    X_middle_step = X
    for i in range(PREDICTION_OUTPUT_LENGTH):
        if i == 0:
            y_predicted = model.predict(X_middle_step)
            y_true = y[:, i, :]
            y_predicted_s_indices = np.argmax(y_predicted, axis=1)
            y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
            y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, EXPERIMENT_PARAMETERS, spatial_index)
            y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, EXPERIMENT_PARAMETERS, spatial_index)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            X_middle_step = X_middle_step[:, 1:, :]
            y_predicted_s_indices = y_predicted_s_indices.reshape(y_predicted_s_indices.shape[0], 1, y_predicted_s_indices.shape[1])
            # print(X_middle_step.shape)
            # print(y_predicted_s_indices.shape)
            X_middle_step = np.concatenate((X_middle_step, y_predicted_s_indices), axis=1)
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = y_predicted_latlon

        elif i == PREDICTION_OUTPUT_LENGTH:
            y_true = y[:, i, :]
            y_predicted = model.predict(X_middle_step)
            # print(y_predicted)
            # print(y_predicted.shape)
            y_predicted_s_indices = np.argmax(y_predicted, axis=1)
            y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
            y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, EXPERIMENT_PARAMETERS, spatial_index)
            y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, EXPERIMENT_PARAMETERS, spatial_index)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)

        else:
            y_predicted = model.predict(X_middle_step)
            y_true = y[:, i, :]
            y_predicted_s_indices = np.argmax(y_predicted, axis=1)
            y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
            y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, EXPERIMENT_PARAMETERS, spatial_index)
            y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, EXPERIMENT_PARAMETERS, spatial_index)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            X_middle_step = X_middle_step[:, 1:, :]
            y_predicted_s_indices = y_predicted_s_indices.reshape(y_predicted_s_indices.shape[0], 1, y_predicted_s_indices.shape[1])
            # print(X_middle_step.shape)
            # print(y_predicted.shape)
            X_middle_step = np.concatenate((X_middle_step, y_predicted_s_indices), axis=1)
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)
    return y_predicted_sequence


def calculate_rmse_in_meters(input_array):
    origin = (input_array[0], input_array[1])
    destination = (input_array[2], input_array[3])
    distance = vincenty(origin, destination).meters
    return distance


def calculate_rmse_on_array(y_predicted, y_test):
    y_join = np.concatenate((y_predicted, y_test), axis=1)
    # print(y_join)
    # print(y_join.shape)
    error_meter = np.apply_along_axis(calculate_rmse_in_meters, axis=1, arr=y_join)
    # print(error_meter.shape)
    rmse_meter_mean = error_meter.mean()
    # logger.info("RMSE in meters: %s" % rmse_meter_mean)
    return rmse_meter_mean


def create_geojson_line_prediction(X, y, outfile, EXPERIMENT_PARAMETERS):
    logger.info("Creating GeoJSON file")
    predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    with open(outfile, 'w', encoding="utf-8") as geojson_file:
        X_shape = X.shape
        if predict_length > 1:
            y_length = y.shape[1]
        else:
            y_length = 1
        my_feature_list = []
        for i in range(X_shape[0]):
            properties = {"new_uid": i}
            line = []
            lat, lon = X[i, -1, -2], X[i, -1, -1]
            line.append((lon, lat))
            if y_length == 1:
                lat, lon = np.asscalar(y[i, -2]), np.asscalar(y[i, -1])
                line.append((lon, lat))
            else:
                for j in range(y_length):
                    lat, lon = np.asscalar(y[i, j, -2]), np.asscalar(y[i, j, -1])
                    line.append((lon, lat))
            my_line = LineString(line)
            my_feature = Feature(geometry=my_line, properties=properties)
            my_feature_list.append(my_feature)
        my_feature_collection = FeatureCollection(my_feature_list)
        dump = geojson.dumps(my_feature_collection, sort_keys=True)
        geojson_file.write(dump)
    return None


def create_geojson_line_observation(X, outfile, EXPERIMENT_PARAMETERS):
    logger.info("Creating GeoJSON file")
    predict_length = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    with open(outfile, 'w', encoding="utf-8") as geojson_file:
        X_shape = X.shape
        my_feature_list = []
        for i in range(X_shape[0]):
            properties = {"new_uid": i}
            line = []
            for j in range(X_shape[1]):
                lat, lon = X[i, j, -2], X[i, j, -1]
                line.append((lon, lat))
            my_line = LineString(line)
            my_feature = Feature(geometry=my_line, properties=properties)
            my_feature_list.append(my_feature)
        my_feature_collection = FeatureCollection(my_feature_list)
        dump = geojson.dumps(my_feature_collection, sort_keys=True)
        geojson_file.write(dump)
    return None


if __name__ == '__main__':
    slack_client = s.sc
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR
    X_GRID_FILE = s.X_GRID_FILE
    Y_GRID_FILE = s.Y_GRID_FILE

    X_FILE = s.X_FILE
    Y_FILE = s.Y_FILE
    Y_FILE_PREDICTED_LSTM = s.Y_FILE_PREDICTED_LSTM
    Y_FILE_PREDICTED_VELOCITY = s.Y_FILE_PREDICTED_VELOCITY

    MODEL_FILE_LSTM_GRID = s.MODEL_FILE_LSTM_GRID
    MODEL_WEIGHT_FILE_LSTM_GRID = s.MODEL_WEIGHT_FILE_LSTM_GRID

    GEOJSON_FILE_OBSERVATION_GRID = s.GEOJSON_FILE_OBSERVATION_GRID
    GEOJSON_FILE_TRUE_GRID = s.GEOJSON_FILE_TRUE_GRID
    GEOJSON_FILE_PREDICTED_LSTM_GRID = s.GEOJSON_FILE_PREDICTED_LSTM_GRID


    spatial_index = preprocessing_gps.define_spatial_index(EXPERIMENT_PARAMETERS)
    max_s_index = spatial_index[2] * spatial_index[3]

    X_train, y_train, X_test, y_test = load_grid_dataset(X_GRID_FILE, Y_GRID_FILE, EXPERIMENT_PARAMETERS)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    # # Train model
    lstm_model = training_rnn_grid(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, max_s_index, MODEL_FILE_LSTM_GRID, MODEL_WEIGHT_FILE_LSTM_GRID, FIGURE_DIR)

    # # Load model
    lstm_model = load_model(MODEL_FILE_LSTM_GRID)

    y_predicted_sequence = prediction_multiple_steps(lstm_model, X_test, y_test, spatial_index, EXPERIMENT_PARAMETERS)
    X_test_latlon = convert_spatial_index_array_to_coordinate_array(X_test, EXPERIMENT_PARAMETERS, spatial_index, Multistep=True)
    y_test_latlon = convert_spatial_index_array_to_coordinate_array(y_test, EXPERIMENT_PARAMETERS, spatial_index, Multistep=True)

    create_geojson_line_observation(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_OBSERVATION_GRID, EXPERIMENT_PARAMETERS)
    create_geojson_line_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_TRUE_GRID, EXPERIMENT_PARAMETERS)
    create_geojson_line_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_predicted_sequence[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_PREDICTED_LSTM_GRID, EXPERIMENT_PARAMETERS)


    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__+" is finished.")
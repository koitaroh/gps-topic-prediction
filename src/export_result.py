import requests
import json
import csv
import os
import time
import random
import sqlalchemy
import geojson
from geojson import LineString, FeatureCollection, Feature
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy
import h5py
from geopy.distance import vincenty
import datetime
from datetime import datetime, timedelta
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
import math
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

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

numpy.random.seed(7)
random.seed(7)



def prediction_lstm(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, MODEL_FILE, MODEL_WEIGHT_FILE_LSTM, FIGURE_DIR):
    '''
    :param X_train: Training data with shape
    :param y_train: The number of feature maps we'd like to calculate
    :param X_test: The filter width
    :param y_test: The stride
    :param EXPERIMENT_PARAMETERS: Experiment parameters
    :return: none
    '''
    # print('Pad sequences (samples x time)')
    # X_train = sequence.pad_sequences(X_train, maxlen=EXPERIMENT_PARAMETERS['RECALL_LENGTH'])
    # X_test = sequence.pad_sequences(X_test, maxlen=EXPERIMENT_PARAMETERS['RECALL_LENGTH'])
    # y_train = sequence.pad_sequences(y_train, maxlen=EXPERIMENT_PARAMETERS['RECALL_LENGTH'])
    # y_test = sequence.pad_sequences(y_test, maxlen=EXPERIMENT_PARAMETERS['RECALL_LENGTH'])
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    # print(X_train)
    # print(X_test)
    # print(y_test)

    input_shape = X_train.shape[1:]
    # print(input_shape)

    in_out_neurons = 2
    hidden_neurons = 128
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=input_shape, return_sequences=True, name='lstm-1', dropout=0.2))
    # model.add(LSTM(hidden_neurons, return_sequences=True, name='lstm-2', dropout=0.2))
    # model.add(LSTM(hidden_neurons, return_sequences=True, name='lstm-3', dropout=0.2))
    model.add(LSTM(hidden_neurons, return_sequences=False, name='lstm-4', dropout=0.2))
    model.add(Dense(in_out_neurons, name='dense-1'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['mae', 'mape'])
    # model.compile(loss="mean_absolute_percentage_error", optimizer="rmsprop", metrics=['mae', 'mse'])
    model.summary()

    history = model.fit(X_train, y_train, batch_size=50, epochs=50, validation_data=(X_test, y_test), callbacks=[es_cb])
    # history = model.fit(X_train, y_train, batch_size=50, epochs=50, validation_data=(X_test, y_test))

    model.save(MODEL_FILE)
    model.save_weights(MODEL_WEIGHT_FILE_LSTM)

    # print(model.metrics_names)
    loss, mae, mape = model.evaluate(X_test, y_test, batch_size=300)
    # print('MSE score:', loss)
    # print('MAE score:', mae)
    # print('MAPE score:', mape)

    logger.info('MSE score: %s' % loss)
    logger.info('MAE score: %s' % mae)
    logger.info('MAPE score: %s' % mape)

    # print(history.history.keys())
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(FIGURE_DIR + "accuracy_mae.png")
    plt.close()
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.title('Mean Absolute Percentage Error')
    plt.ylabel('MAPE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(FIGURE_DIR + "accuracy_mape.png")
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss (Mean Squared Error)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(FIGURE_DIR + "accuracy_mse.png")
    plt.close()


    y_predicted = model.predict(X_test)
    # result = numpy.concatenate((X_test, y_test, y_predicted), axis=3)
    # print(result)
    # print(result.shape)

    # print(numpy.sqrt(((predicted - y_test) ** 2).mean(axis=0)).mean())  # Printing RMSE
    return y_predicted


def prediction_gru():
    pass


def prediction_simplernn():
    pass


def prediction_hmm():
    pass


def prediction_velocity(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS):
    x_last_two = X_test[:, -2:, -2:]
    last_diff = numpy.diff(x_last_two, axis=1)
    x_last = X_test[:, -1:, -2:]
    prediction = numpy.add(last_diff, x_last)
    prediction_shape = prediction.shape
    prediction = prediction.reshape(prediction_shape[0], prediction_shape[2])
    return prediction


def prediction_particle_filter():
    pass


def calculate_rmse_in_meters(input_array):
    origin = (input_array[0], input_array[1])
    destination = (input_array[2], input_array[3])
    distance = vincenty(origin, destination).meters
    return  distance


def calculate_rmse_on_array(y_predicted, y_test):
    y_join = numpy.concatenate((y_predicted, y_test), axis=1)
    # print(y_join)
    # print(y_join.shape)
    error_meter = numpy.apply_along_axis(calculate_rmse_in_meters, axis=1, arr=y_join)
    # print(error_meter.shape)
    rmse_meter_mean = error_meter.mean()
    logger.info("RMSE in meters: %s" % rmse_meter_mean)
    return None


def create_mobmap_csv(X, y, outfile, EXPERIMENT_PARAMETERS):
    logger.info("Creating GeoJSON file")
    X_shape = X.shape
    recall_length = EXPERIMENT_PARAMETERS['RECALL_LENGTH']
    predict_length = EXPERIMENT_PARAMETERS['PREDICT_LENGTH']
    dtype = {'new_uid': 'int', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}
    pandas.DataFrame()

    return None


def create_geojson_line(X, y, outfile, EXPERIMENT_PARAMETERS):
    logger.info("Creating GeoJSON file")
    predict_length = EXPERIMENT_PARAMETERS['PREDICT_LENGTH']
    with open(outfile, 'w', encoding="utf-8") as geojson_file:
        X_shape = X.shape
        # print(X.shape)
        # print(y.shape)
        x_length = X.shape[1]
        if predict_length > 1:
            y_length = y.shape[1]
        else:
            y_length = 1
        # print(x_length)
        # print(y_length)
        my_feature_list = []
        for i in range(X_shape[0]):
            properties = {"new_uid": i}
            line = []
            for j in range(X_shape[1]):
                # print(X[i, j, -2:])
                lat, lon = X[i, j, -2], X[i, j, -1]
                # print(lat, lon)
                line.append((lon, lat))
            if y_length == 1:
                # print(y[i])
                lat, lon = numpy.asscalar(y[i, -2]), numpy.asscalar(y[i, -1])
                # print(lat, lon)
                line.append((lon, lat))
            # print(line)
            my_line = LineString(line)
            my_feature = Feature(geometry=my_line, properties=properties)
            my_feature_list.append(my_feature)
        my_feature_collection = FeatureCollection(my_feature_list)
        dump = geojson.dumps(my_feature_collection, sort_keys=True)
        geojson_file.write(dump)
    return None


def create_geojson_prediction_line(X, y, outfile, EXPERIMENT_PARAMETERS):
    logger.info("Creating Prediction GeoJSON file")
    predict_length = EXPERIMENT_PARAMETERS['PREDICT_LENGTH']
    with open(outfile, 'w', encoding="utf-8") as geojson_file:
        X_shape = X.shape
        # print(X.shape)
        # print(y.shape)
        x_length = X.shape[1]
        if predict_length > 1:
            y_length = y.shape[1]
        else:
            y_length = 1
        # print(x_length)
        # print(y_length)
        my_feature_list = []
        for i in range(X_shape[0]):
            properties = {"new_uid": i}
            line = []
            lat, lon = X[i, -1, -2], X[i, -1, -1]
            # print(lat, lon)
            line.append((lon, lat))
            if y_length == 1:
                # print(y[i])
                lat, lon = numpy.asscalar(y[i, -2]), numpy.asscalar(y[i, -1])
                # print(lat, lon)
                line.append((lon, lat))
            # print(line)
            my_line = LineString(line)
            my_feature = Feature(geometry=my_line, properties=properties)
            my_feature_list.append(my_feature)
        my_feature_collection = FeatureCollection(my_feature_list)
        dump = geojson.dumps(my_feature_collection, sort_keys=True)
        geojson_file.write(dump)
    return None



def save_numpy_array(input_array, output_file):
    numpy.save(file=output_file, arr=input_array)
    return None



if __name__ == '__main__':
    TEST_CSV = s.TEST_CSV
    TEST_CSV_FILTERED = s.TEST_CSV_FILTERED
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR
    X_FILE = s.X_FILE
    Y_FILE = s.Y_FILE
    Y_FILE_PREDICTED_LSTM = s.Y_FILE_PREDICTED_LSTM
    Y_FILE_PREDICTED_VELOCITY = s.Y_FILE_PREDICTED_VELOCITY

    MODEL_FILE_LSTM = s.MODEL_FILE_LSTM
    MODEL_WEIGHT_FILE_LSTM = s.MODEL_WEIGHT_FILE_LSTM

    GEOJSON_FILE_TURE = s.GEOJSON_FILE_TRUE
    GEOJSON_FILE_PREDICTED_LSTM = s.GEOJSON_FILE_PREDICTED_LSTM
    GEOJSON_FILE_PREDICTED_VELOCITY = s.GEOJSON_FILE_PREDICTED_VELOCITY

    (X_train, y_train), (X_test, y_test), (X_scaler, y_scaler) = load_dataset.load_dataset(TEST_CSV_FILTERED, EXPERIMENT_PARAMETERS)

    y_predicted_lstm = prediction_lstm(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, MODEL_FILE_LSTM, MODEL_WEIGHT_FILE_LSTM, FIGURE_DIR)
    y_predicted_velocity = prediction_velocity(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS)

    X_test = load_dataset.inverse_scale_transform_sample(X_test, X_scaler, X=True)
    y_test = load_dataset.inverse_scale_transform_sample(y_test, y_scaler)
    y_predicted_lstm = load_dataset.inverse_scale_transform_sample(y_predicted_lstm, y_scaler)
    y_predicted_velocity = load_dataset.inverse_scale_transform_sample(y_predicted_velocity, y_scaler)

    logger.info("Accuracy of LSTM")
    calculate_rmse_on_array(y_predicted_lstm, y_test)
    logger.info("Accuracy of velocity")
    calculate_rmse_on_array(y_predicted_velocity, y_test)

    save_numpy_array(X_test, X_FILE)
    save_numpy_array(y_test, Y_FILE)
    save_numpy_array(y_predicted_lstm, Y_FILE_PREDICTED_LSTM)
    save_numpy_array(y_predicted_velocity, Y_FILE_PREDICTED_VELOCITY)


    X_test = numpy.load(X_FILE)
    y_test = numpy.load(Y_FILE)
    y_predicted_lstm = numpy.load(Y_FILE_PREDICTED_LSTM)
    y_predicted_velocity = numpy.load(Y_FILE_PREDICTED_VELOCITY)

    create_geojson_line(X_test, y_test, GEOJSON_FILE_TURE, EXPERIMENT_PARAMETERS)
    create_geojson_prediction_line(X_test, y_predicted_lstm, GEOJSON_FILE_PREDICTED_LSTM, EXPERIMENT_PARAMETERS)
    create_geojson_prediction_line(X_test, y_predicted_velocity, GEOJSON_FILE_PREDICTED_VELOCITY, EXPERIMENT_PARAMETERS)
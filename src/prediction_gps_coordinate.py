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
import numpy as np
import h5py
from geopy.distance import vincenty
from datetime import datetime, timedelta
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
import math
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

from project.references import settings as s
import load_dataset

np.random.seed(7)
random.seed(7)

def load_coordinates_dataset(X_COORDINATE_FILE, Y_COORDINATE_FILE, EXPERIMENT_PARAMETERS):
    X_all, y_all, scaler = load_dataset.load_coordinates_numpy_input_file(X_COORDINATE_FILE, Y_COORDINATE_FILE)
    (X_train, y_train), (X_test, y_test) = load_dataset.devide_sample(X_all, y_all, scaler, EXPERIMENT_PARAMETERS)
    X_train, y_train = load_dataset.create_full_training_sample(X_train, y_train)
    return (X_train, y_train), (X_test, y_test), (scaler)


def training_lstm(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, MODEL_FILE, MODEL_WEIGHT_FILE_LSTM, FIGURE_DIR):
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
    y_test_one_step = y_test[:,0,:]
    print('y_test shape:', y_test_one_step.shape)
    # print(X_train)
    # print(X_test)
    # print(y_test)

    input_shape = X_train.shape[1:]
    # print(input_shape)

    in_out_neurons = 2
    hidden_neurons = 128
    batch_size = 50
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=input_shape, return_sequences=True, name='lstm-1'))
    model.add(Dropout(0.2))
    # model.add(LSTM(hidden_neurons, return_sequences=True, name='lstm-2', dropout=0.2))
    # model.add(LSTM(hidden_neurons, return_sequences=True, name='lstm-3', dropout=0.2))
    model.add(LSTM(hidden_neurons, return_sequences=False, name='lstm-4'))
    model.add(Dropout(0.2))
    model.add(Dense(in_out_neurons, name='dense-1'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['mae', 'mape'])
    # model.compile(loss="mean_absolute_percentage_error", optimizer="rmsprop", metrics=['mae', 'mse'])
    model.summary()


    ### Functional API
    # x_input = Input(shape=input_shape)
    # lstm_1 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-1')(x_input)
    # lstm_2 = LSTM(hidden_neurons, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='lstm-2')(lstm_1)
    # main_output = Dense(in_out_neurons, name='dense-1')(lstm_2)
    # model = Model(inputs=[x_input], outputs=[main_output])
    # model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae', 'mape'])
    # model.summary()

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=50, validation_data=(X_test, y_test_one_step), callbacks=[es_cb])
    # history = model.fit(X_train, y_train, batch_size=50, epochs=50, validation_data=(X_test, y_test))

    model.save(MODEL_FILE)
    model.save_weights(MODEL_WEIGHT_FILE_LSTM)

    # print(model.metrics_names)
    loss, mae, mape = model.evaluate(X_test, y_test_one_step, batch_size=300)
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

    return model


def prediction_one_step(model, X):
    y_predicted = model.predict(X)
    return y_predicted


def prediction_multiple_steps(model, X, y, scaler, EXPERIMENT_PARAMETERS):
    PREDICTION_OUTPUT_LENGTH = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    X_middle_step = X
    for i in range(PREDICTION_OUTPUT_LENGTH):
        if i == 0:
            y_predicted = prediction_one_step(model, X_middle_step)
            y_true = y[:, i, :]
            y_predicted_latlon = load_dataset.inverse_scale_transform_sample(y_predicted, scaler)
            y_predicted_velocity = prediction_velocity(X, i)
            y_predicted_velocity = load_dataset.inverse_scale_transform_sample(y_predicted_velocity, scaler)
            y_true = load_dataset.inverse_scale_transform_sample(y_true, scaler)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_velocity, y_true)
            logger.info("Baseline velocity accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            X_middle_step = X_middle_step[:, 1:, :]
            y_predicted = y_predicted.reshape(y_predicted.shape[0], 1, y_predicted.shape[1])
            X_middle_step = np.concatenate((X_middle_step, y_predicted), axis=1)
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = y_predicted_latlon

        elif i == PREDICTION_OUTPUT_LENGTH:
            y_predicted = prediction_one_step(model, X_middle_step)
            y_true = y[:,i,:]
            y_predicted_latlon = load_dataset.inverse_scale_transform_sample(y_predicted, scaler)
            y_predicted_velocity = prediction_velocity(X, i)
            y_predicted_velocity = load_dataset.inverse_scale_transform_sample(y_predicted_velocity, scaler)
            y_true = load_dataset.inverse_scale_transform_sample(y_true, scaler)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true)
            logger.info("Model accuracy in timestep %s: %s" % (str(i+1), rmse_meter_mean))
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_velocity, y_true)
            logger.info("Baseline velocity accuracy in timestep %s: %s" % (str(i+1), rmse_meter_mean))
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)

        else:
            y_predicted = prediction_one_step(model, X_middle_step)
            y_true = y[:, i, :]
            y_predicted_latlon = load_dataset.inverse_scale_transform_sample(y_predicted, scaler)
            y_predicted_velocity = prediction_velocity(X, i)
            y_predicted_velocity = load_dataset.inverse_scale_transform_sample(y_predicted_velocity, scaler)
            y_true = load_dataset.inverse_scale_transform_sample(y_true, scaler)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_velocity, y_true)
            logger.info("Baseline velocity accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            X_middle_step = X_middle_step[:, 1:, :]
            y_predicted = y_predicted.reshape(y_predicted.shape[0], 1, y_predicted.shape[1])
            X_middle_step = np.concatenate((X_middle_step, y_predicted), axis=1)
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)
    return y_predicted_sequence


def prediction_velocity(X, time_step):
    time_step += 1
    x_last_two = X[:, -2:, -2:]
    last_diff = np.diff(x_last_two, axis=1)
    x_last = X[:, -1:, -2:]
    last_diff *= time_step + 1
    prediction = np.add(last_diff, x_last)
    prediction_shape = prediction.shape
    prediction = prediction.reshape(prediction_shape[0], prediction_shape[2])
    return prediction


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


def create_mobmap_csv(X, y, outfile, EXPERIMENT_PARAMETERS):
    logger.info("Creating GeoJSON file")
    X_shape = X.shape
    recall_length = EXPERIMENT_PARAMETERS['RECALL_LENGTH']
    predict_length = EXPERIMENT_PARAMETERS['PREDICT_LENGTH']
    dtype = {'new_uid': 'int', 'timestamp': 'str', 'y': 'float64', 'x': 'float64'}
    pandas.DataFrame()
    return None


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
    X_COORDINATE_FILE = s.X_COORDINATE_FILE
    Y_COORDINATE_FILE = s.Y_COORDINATE_FILE
    X_GRID_FILE = s.X_GRID_FILE
    Y_GRID_FILE = s.Y_GRID_FILE

    X_FILE = s.X_FILE
    Y_FILE = s.Y_FILE
    Y_FILE_PREDICTED_LSTM = s.Y_FILE_PREDICTED_LSTM
    Y_FILE_PREDICTED_VELOCITY = s.Y_FILE_PREDICTED_VELOCITY

    MODEL_FILE_LSTM = s.MODEL_FILE_LSTM
    MODEL_WEIGHT_FILE_LSTM = s.MODEL_WEIGHT_FILE_LSTM

    GEOJSON_FILE_OBSERVATION = s.GEOJSON_FILE_OBSERVATION
    GEOJSON_FILE_TURE = s.GEOJSON_FILE_TRUE
    GEOJSON_FILE_PREDICTED_LSTM = s.GEOJSON_FILE_PREDICTED_LSTM

    (X_train, y_train), (X_test, y_test), (scaler) = load_coordinates_dataset(X_COORDINATE_FILE, Y_COORDINATE_FILE, EXPERIMENT_PARAMETERS)

    # Train model
    lstm_model = training_lstm(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, MODEL_FILE_LSTM, MODEL_WEIGHT_FILE_LSTM, FIGURE_DIR)

    # Load model
    lstm_model = load_model(MODEL_FILE_LSTM)

    y_predicted_sequence = prediction_multiple_steps(lstm_model, X_test, y_test, scaler, EXPERIMENT_PARAMETERS)
    X_test = load_dataset.inverse_scale_transform_sample(X_test, scaler, Multistep=True)
    y_test = load_dataset.inverse_scale_transform_sample(y_test, scaler, Multistep=True)

    create_geojson_line_observation(X_test[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_OBSERVATION, EXPERIMENT_PARAMETERS)
    create_geojson_line_prediction(X_test[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_test[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_TURE, EXPERIMENT_PARAMETERS)
    create_geojson_line_prediction(X_test[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_predicted_sequence[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_PREDICTED_LSTM, EXPERIMENT_PARAMETERS)

    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__+" is finished.")
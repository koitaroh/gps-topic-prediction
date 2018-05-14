import random
import geojson
from geojson import LineString, FeatureCollection, Feature
import numpy as np
from geopy.distance import vincenty
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.models import load_model, Model

import matplotlib
import random

import geojson
import matplotlib
import numpy as np
from geojson import LineString, FeatureCollection, Feature
from geopy.distance import vincenty
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
# from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sklearn.preprocessing import StandardScaler, LabelBinarizer

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Logging ver. 2017-10-30
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

from project.references import settings as s
import load_dataset

np.random.seed(7)
random.seed(7)

def load_coordinates_mode_dataset(X_COORDINATE_FILE, Y_COORDINATE_FILE, X_MODE_FILE, Y_MODE_FILE, EXPERIMENT_PARAMETERS):
    X_all, y_all, scaler, X_mode_all, y_mode_all, lb = load_coordinates_mode_numpy_input_file(X_COORDINATE_FILE, Y_COORDINATE_FILE, X_MODE_FILE, Y_MODE_FILE)
    X_train, y_train, X_test, y_test, X_mode_train, X_mode_test, y_mode_train, y_mode_test = devide_sample(X_all, y_all, scaler, X_mode_all, y_mode_all, lb, EXPERIMENT_PARAMETERS)
    X_train, y_train = load_dataset.create_full_training_sample(X_train, y_train)
    X_mode_train, y_mode_train = load_dataset.create_full_training_sample(X_mode_train, y_mode_train)
    return X_train, y_train, X_test, y_test, X_mode_train, X_mode_test, y_mode_train, y_mode_test, scaler, lb


def load_coordinates_mode_numpy_input_file(X_all, y_all, X_mode_all, y_mode_all):
    X_all = np.load(X_all)
    y_all = np.load(y_all)
    X_mode_all = np.load(X_mode_all)
    y_mode_all = np.load(y_mode_all)
    # print(X_all)
    # print(y_all)
    # print(X_mode_all)
    # print(y_mode_all)

    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    x_shape = X_all.shape
    y_shape = y_all.shape
    X_all = X_all.reshape(x_shape[0] * x_shape[1], x_shape[2])
    y_all = y_all.reshape(y_shape[0] * y_shape[1], y_shape[2])
    X_y_all = np.concatenate((X_all, y_all), axis=0)
    # y_one_step = y_all[:,0,:]
    scaler.fit(X_y_all)
    X_all = X_all.reshape(x_shape[0], x_shape[1], x_shape[2])
    y_all = y_all.reshape(y_shape[0], y_shape[1], y_shape[2])

    ## Binalize mode
    X_mode_all = X_mode_all.reshape(x_shape[0] * x_shape[1], 1)
    y_mode_all = y_mode_all.reshape(y_shape[0] * y_shape[1], 1)
    X_y_mode_all = np.concatenate((X_mode_all, y_mode_all), axis=0)
    lb = LabelBinarizer()
    lb.fit(X_y_mode_all)
    print(lb.classes_)
    print(len(lb.classes_))
    X_mode_all = X_mode_all.reshape(x_shape[0], x_shape[1], 1)
    y_mode_all = y_mode_all.reshape(y_shape[0], y_shape[1], 1)

    return X_all, y_all, scaler, X_mode_all, y_mode_all, lb


def devide_sample(X_all, y_all, scaler, X_mode_all, y_mode_all, lb, EXPERIMENT_PARAMETERS):
    X_all = scale_transform_sample(X_all, scaler, Multistep=True)
    y_all = scale_transform_sample(y_all, scaler, Multistep=True)

    x_shape = X_all.shape
    y_shape = y_all.shape
    num_mode = len(lb.classes_)

    # print(X_mode_all)
    X_mode_all = X_mode_all.reshape(x_shape[0] * x_shape[1], 1)
    y_mode_all = y_mode_all.reshape(y_shape[0] * y_shape[1], 1)
    X_mode_all = lb.transform(X_mode_all)
    y_mode_all = lb.transform(y_mode_all)
    X_mode_all = X_mode_all.reshape(x_shape[0], x_shape[1], num_mode)
    y_mode_all = y_mode_all.reshape(y_shape[0], y_shape[1], num_mode)
    # print(X_mode_all)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, test_size=0.2, random_state=7)
    X_mode_train, X_mode_test, y_mode_train, y_mode_test = train_test_split(X_mode_all, y_mode_all, train_size=0.8, test_size=0.2, random_state=7)

    return X_train, y_train, X_test, y_test, X_mode_train, X_mode_test, y_mode_train, y_mode_test


def scale_transform_sample(input_array, scaler, Multistep=False):
    if Multistep:
        X_shape = input_array.shape
        # print(X_shape)
        input_array = input_array.reshape(X_shape[0] * X_shape[1], X_shape[2])
        output_array = scaler.transform(input_array)
        output_array = output_array.reshape(X_shape[0], X_shape[1], X_shape[2])
    else:
        output_array = scaler.transform(input_array)
    return output_array


def inverse_scale_transform_sample(input_array, scaler, Multistep=False):
    if Multistep:
        X_shape = input_array.shape
        # print(X_shape)
        input_array = input_array.reshape(X_shape[0] * X_shape[1], X_shape[2])
        output_array = scaler.inverse_transform(input_array)
        output_array = output_array.reshape(X_shape[0], X_shape[1], X_shape[2])
    else:
        output_array = scaler.inverse_transform(input_array)
    return output_array


def create_full_training_sample(X_train_original, y_train_original):
    PREDICTION_OUTPUT_LENGTH = y_train_original.shape[1]
    for i in range(PREDICTION_OUTPUT_LENGTH):
        if i == 0:
            X_train_all = X_train_original
            y_train_all = y_train_original[:, i, :]
        else:
            X = X_train_original[:, i:, :]
            X = np.concatenate((X, y_train_original[:, 0:i, :]), axis=1)
            y = y_train_original[:, i, :]
            X_train_all = np.concatenate((X_train_all, X), axis=0)
            y_train_all = np.concatenate((y_train_all, y), axis=0)
    return X_train_all, y_train_all


def training_lstm(X_train, y_train, X_test, y_test, X_mode_train, X_mode_test, y_mode_train, y_mode_test, EXPERIMENT_PARAMETERS, MODEL_FILE, MODEL_WEIGHT_FILE_LSTM, FIGURE_DIR):

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    y_test_one_step = y_test[:,0,:]
    print('y_test shape:', y_test_one_step.shape)

    input_shape = X_train.shape[1:]
    input_mode_shape = X_mode_train.shape[1:]
    print(input_shape)
    print(input_mode_shape)

    in_out_neurons = 2
    hidden_neurons = 128
    batch_size = 50
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    x_input = Input(shape=input_shape)
    lstm_x_1 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-x-1')(x_input)

    x_mode_input = Input(shape=input_mode_shape)
    lstm_mode_1 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-mode-1')(x_mode_input)

    lstm_input = Concatenate()([lstm_x_1, lstm_mode_1])

    lstm_1 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-1')(lstm_input)
    lstm_2 = LSTM(hidden_neurons, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='lstm-2')(lstm_1)
    main_output = Dense(in_out_neurons, name='dense-1')(lstm_2)

    model = Model(inputs=[x_input, x_mode_input], outputs=[main_output])
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae', 'mape'])
    model.summary()


    history = model.fit([X_train, X_mode_train], [y_train], batch_size=batch_size, epochs=50, validation_data=([X_test, X_mode_test], y_test_one_step), callbacks=[es_cb])

    model.save(MODEL_FILE)
    model.save_weights(MODEL_WEIGHT_FILE_LSTM)
    loss, mae, mape = model.evaluate([X_test, X_mode_test], y_test_one_step, batch_size=batch_size)
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


def prediction_one_step(model, X, X_mode):
    y_predicted = model.predict([X, X_mode])
    return y_predicted


def prediction_multiple_steps(model, X, y, X_mode, y_mode, scaler, EXPERIMENT_PARAMETERS):
    PREDICTION_OUTPUT_LENGTH = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    X_middle_step = X
    X_mode_middle_step = X_mode

    for i in range(PREDICTION_OUTPUT_LENGTH):
        if i == 0:
            y_predicted = prediction_one_step(model, X_middle_step, X_mode_middle_step)
            y_true = y[:, i, :]
            y_mode_true = y_mode[:, i, :]
            y_predicted_latlon = load_dataset.inverse_scale_transform_sample(y_predicted, scaler)
            y_predicted_velocity = prediction_velocity(X, i)
            y_predicted_velocity = load_dataset.inverse_scale_transform_sample(y_predicted_velocity, scaler)
            y_true = load_dataset.inverse_scale_transform_sample(y_true, scaler)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_velocity, y_true)
            logger.info("Baseline velocity accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            X_middle_step = X_middle_step[:, 1:, :]
            X_mode_middle_step = X_mode_middle_step[:, 1:, :]
            y_predicted = y_predicted.reshape(y_predicted.shape[0], 1, y_predicted.shape[1])
            y_mode_true = y_mode_true.reshape(y_mode_true.shape[0], 1, y_mode_true.shape[1])
            X_middle_step = np.concatenate((X_middle_step, y_predicted), axis=1)
            X_mode_middle_step = np.concatenate((X_mode_middle_step, y_mode_true), axis=1)
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = y_predicted_latlon

        elif i == PREDICTION_OUTPUT_LENGTH:
            y_predicted = prediction_one_step(model, X_middle_step, X_mode_middle_step)
            y_true = y[:, i, :]
            y_mode_true = y_mode[:, i, :]
            y_predicted_latlon = load_dataset.inverse_scale_transform_sample(y_predicted, scaler)
            y_predicted_velocity = prediction_velocity(X, i)
            y_predicted_velocity = load_dataset.inverse_scale_transform_sample(y_predicted_velocity, scaler)
            y_true = load_dataset.inverse_scale_transform_sample(y_true, scaler)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_velocity, y_true)
            logger.info("Baseline velocity accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)

        else:
            y_predicted = prediction_one_step(model, X_middle_step, X_mode_middle_step)
            y_true = y[:, i, :]
            y_mode_true = y_mode[:, i, :]
            y_predicted_latlon = load_dataset.inverse_scale_transform_sample(y_predicted, scaler)
            y_predicted_velocity = prediction_velocity(X, i)
            y_predicted_velocity = load_dataset.inverse_scale_transform_sample(y_predicted_velocity, scaler)
            y_true = load_dataset.inverse_scale_transform_sample(y_true, scaler)
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_latlon, y_true)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            rmse_meter_mean = calculate_rmse_on_array(y_predicted_velocity, y_true)
            logger.info("Baseline velocity accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            X_middle_step = X_middle_step[:, 1:, :]
            X_mode_middle_step = X_mode_middle_step[:, 1:, :]
            y_predicted = y_predicted.reshape(y_predicted.shape[0], 1, y_predicted.shape[1])
            y_mode_true = y_mode_true.reshape(y_mode_true.shape[0], 1, y_mode_true.shape[1])
            X_middle_step = np.concatenate((X_middle_step, y_predicted), axis=1)
            X_mode_middle_step = np.concatenate((X_mode_middle_step, y_mode_true), axis=1)
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
    X_MODE_FILE = s.X_MODE_FILE
    Y_MODE_FILE = s.Y_MODE_FILE

    X_FILE = s.X_FILE
    Y_FILE = s.Y_FILE
    Y_FILE_PREDICTED_LSTM = s.Y_FILE_PREDICTED_LSTM
    Y_FILE_PREDICTED_VELOCITY = s.Y_FILE_PREDICTED_VELOCITY

    MODEL_FILE_LSTM = s.MODEL_FILE_LSTM
    MODEL_WEIGHT_FILE_LSTM = s.MODEL_WEIGHT_FILE_LSTM

    GEOJSON_FILE_OBSERVATION = s.GEOJSON_FILE_OBSERVATION
    GEOJSON_FILE_TURE = s.GEOJSON_FILE_TRUE
    GEOJSON_FILE_PREDICTED_LSTM = s.GEOJSON_FILE_PREDICTED_LSTM


    X_train, y_train, X_test, y_test, X_mode_train, X_mode_test, y_mode_train, y_mode_test, scaler, lb = load_coordinates_mode_dataset(X_COORDINATE_FILE, Y_COORDINATE_FILE, X_MODE_FILE, Y_MODE_FILE, EXPERIMENT_PARAMETERS)

    # Train model
    lstm_model = training_lstm(X_train, y_train, X_test, y_test, X_mode_train, X_mode_test, y_mode_train, y_mode_test, EXPERIMENT_PARAMETERS, MODEL_FILE_LSTM, MODEL_WEIGHT_FILE_LSTM, FIGURE_DIR)

    # Load model
    lstm_model = load_model(MODEL_FILE_LSTM)

    y_predicted_sequence = prediction_multiple_steps(lstm_model, X_test, y_test, X_mode_test, y_mode_test, scaler, EXPERIMENT_PARAMETERS)
    X_test = load_dataset.inverse_scale_transform_sample(X_test, scaler, Multistep=True)
    y_test = load_dataset.inverse_scale_transform_sample(y_test, scaler, Multistep=True)

    create_geojson_line_observation(X_test[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_OBSERVATION, EXPERIMENT_PARAMETERS)
    create_geojson_line_prediction(X_test[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_test[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_TURE, EXPERIMENT_PARAMETERS)
    create_geojson_line_prediction(X_test[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_predicted_sequence[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_PREDICTED_LSTM, EXPERIMENT_PARAMETERS)

    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__+" is finished.")
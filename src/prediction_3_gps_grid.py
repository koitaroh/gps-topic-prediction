import atexit
import random
import geojson
import matplotlib
import numpy as np
from geojson import Feature
from geojson import FeatureCollection
from geojson import LineString
from keras.callbacks import EarlyStopping
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datetime import datetime
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Flatten, Concatenate, Attention
from tensorflow.keras.models import Model

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Logging ver. 2017-12-14
import logging
from logging import handlers
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
import load_dataset
import utility_spatiotemporal_index

np.random.seed(1)
random.seed(1)

def load_grid_dataset(X_GRID_FILE, Y_GRID_FILE, le_grid, EXPERIMENT_PARAMETERS):
    X_all = np.load(X_GRID_FILE, allow_pickle=True)
    y_all = np.load(Y_GRID_FILE, allow_pickle=True)
    X_all_shape = X_all.shape
    y_all_shape = y_all.shape
    logger.info(f"X_all_shape: {X_all_shape}")
    logger.info(f"y_all_shape: {y_all_shape}")
    X_all = X_all.reshape(X_all_shape[0] * X_all_shape[1], 1)
    y_all = y_all.reshape(y_all_shape[0] * y_all_shape[1], 1)
    X_all = le_grid.transform(X_all)
    y_all = le_grid.transform(y_all)
    X_all = X_all.reshape(X_all_shape[0], X_all_shape[1], 1)
    y_all = y_all.reshape(y_all_shape[0], y_all_shape[1], 1)




    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, test_size=0.2, random_state=7)

    X_train_all, y_train_all = load_dataset.create_full_training_sample(X_train, y_train)

    return X_train_all, y_train_all, X_test, y_test


def convert_spatial_index_array_to_coordinate_array(input_array, le_grid, Multistep=False):
    if Multistep:
        X_shape = input_array.shape
        # print(X_shape) # (18, 12, 1)
        input_array = input_array.reshape(X_shape[0] * X_shape[1], 1)
        input_array = le_grid.inverse_transform(input_array)
        input_array = input_array.reshape(X_shape)
        x_array = np.apply_along_axis(np.vectorize(utility_spatiotemporal_index.convert_spatial_index_to_longitude), 0, input_array)
        y_array = np.apply_along_axis(np.vectorize(utility_spatiotemporal_index.convert_spatial_index_to_latitude), 0, input_array)
        latlon_array = np.concatenate((y_array, x_array), axis=2)
    else:
        X_shape = input_array.shape
        # print(X_shape)  # (18, 12, 1)
        # print(input_array)
        input_array = le_grid.inverse_transform(input_array)
        # print(input_array)
        x_array = np.apply_along_axis(np.vectorize(utility_spatiotemporal_index.convert_spatial_index_to_longitude), 0, input_array)
        y_array = np.apply_along_axis(np.vectorize(utility_spatiotemporal_index.convert_spatial_index_to_latitude), 0, input_array)
        # x_array = x_array.reshape(-1, 1)
        # y_array = y_array.reshape(-1, 1)
        # latlon_array = np.concatenate((y_array, x_array), axis=1)
        latlon_array = np.column_stack((y_array, x_array))
    return latlon_array


def training_lstm_grid(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, max_s_index, MODEL_FILE, MODEL_WEIGHT_FILE_LSTM_GRID, FIGURE_DIR):
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

    model = create_model_lstm_grid(input_shape, max_s_index)
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.summary()
    history = model.fit([X_train], y_train, batch_size=50, epochs=100, validation_data=([X_test], y_test_one_step), callbacks=[es_cb])
    # history = model.fit(X_train, y_train, batch_size=50, epochs=50, validation_data=(X_test, y_test))

    model.save(str(MODEL_FILE))
    model.save_weights(str(MODEL_WEIGHT_FILE_LSTM_GRID))

    loss, accuracy, sparse_top_k_accuracy = model.evaluate([X_test], y_test_one_step, batch_size=300)


    logger.info('sparse_categorical_crossentropy score: %s' % loss)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss (sparse_categorical_crossentropy)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(FIGURE_DIR / "accuracy_sparse_categorical_crossentropy.png")
    plt.close()

    logger.info("sparse_categorical_accuracy: %s" % accuracy)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy (sparse_categorical_accuracy)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(FIGURE_DIR / "accuracy_acc.png")
    plt.close()
    return model


def create_model_lstm_grid(input_shape, max_s_index):
    hidden_neurons = 128
    x_input = Input(shape=input_shape)
    x_input_flatten = Flatten()(x_input)
    x_input_embedding = Embedding(input_dim=max_s_index, output_dim=500, input_length=X_train.shape[1])(x_input_flatten)

    lstm_input = x_input_embedding

    # Two layers 
    # lstm_1 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, activation='softsign', name='lstm-1')(lstm_input)
    # lstm_2 = LSTM(hidden_neurons, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, activation='softsign', name='lstm-2')(lstm_1)
    # main_output = Dense(max_s_index, name='dense-1', activation='softmax')(lstm_2)

    # DeepMove
    lstm_1 = LSTM(hidden_neurons, return_sequences=True, activation='tanh', name='lstm-1')(lstm_input)
    lstm_2 = LSTM(hidden_neurons, return_sequences=True, activation='tanh', name='lstm-2')(lstm_1)
    lstm_3 = LSTM(hidden_neurons, return_sequences=True, activation='tanh', name='lstm-3')(lstm_2)
    attention_output = Attention()([lstm_3, lstm_3])
    lstm_4 = LSTM(hidden_neurons, return_sequences=False, activation='tanh', name='lstm-4')(attention_output)
    main_output = Dense(max_s_index, name='dense-1', activation='softmax')(lstm_4)

    # Four layers
    # lstm_1 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-1')(lstm_input)
    # lstm_2 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-2')(lstm_1)
    # lstm_3 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-3')(lstm_2)
    # lstm_4 = LSTM(hidden_neurons, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='lstm-4')(lstm_3)
    # main_output = Dense(max_s_index, name='dense-1', activation='softmax')(lstm_4)

    # Six layers
    # lstm_1 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-1')(lstm_input)
    # lstm_2 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-2')(lstm_1)
    # lstm_3 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-3')(lstm_2)
    # lstm_4 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-4')(lstm_3)
    # lstm_5 = LSTM(hidden_neurons, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm-5')(lstm_4)
    # lstm_6 = LSTM(hidden_neurons, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='lstm-6')(lstm_5)
    # main_output = Dense(max_s_index, name='dense-1', activation='softmax')(lstm_6)

    model = Model(inputs=[x_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])
    return model


def prediction_multiple_steps_lstm_grid(model, X, y, le_grid, EXPERIMENT_PARAMETERS):
    PREDICTION_OUTPUT_LENGTH = EXPERIMENT_PARAMETERS['PREDICTION_OUTPUT_LENGTH']
    X_middle_step = X

    max_s_index = len(le_grid.classes_)
    for i in tqdm(range(PREDICTION_OUTPUT_LENGTH)):
        if i == 0:
            y_predicted = model.predict([X_middle_step])
            y_true = y[:, i, :]
            y_predicted_s_indices = np.argmax(y_predicted, axis=1)  # For selecting max
            # y_predicted_s_indices = np.array([*map(lambda p: np.random.choice(max_s_index, p=p), y_predicted)]) # For selecting as random sample
            y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
            y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, le_grid)
            y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, le_grid)
            rmse_meter_mean, error_meter_series = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            X_middle_step = X_middle_step[:, 1:, :]
            y_predicted_s_indices = y_predicted_s_indices.reshape(y_predicted_s_indices.shape[0], 1, y_predicted_s_indices.shape[1])
            X_middle_step = np.concatenate((X_middle_step, y_predicted_s_indices), axis=1)
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = y_predicted_latlon

        elif i == PREDICTION_OUTPUT_LENGTH:
            y_true = y[:, i, :]
            y_predicted = model.predict([X_middle_step])
            y_predicted_s_indices = np.argmax(y_predicted, axis=1) # For selecting max
            # y_predicted_s_indices = np.array([*map(lambda p: np.random.choice(max_s_index, p=p), y_predicted)])  # For selecting as random sample
            y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
            y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, le_grid)
            y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, le_grid)
            rmse_meter_mean, error_meter_series = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)

        else:
            y_predicted = model.predict([X_middle_step])
            y_true = y[:, i, :]
            y_predicted_s_indices = np.argmax(y_predicted, axis=1) # For selecting max
            # y_predicted_s_indices = np.array([*map(lambda p: np.random.choice(max_s_index, p=p), y_predicted)])  # For selecting as random sample
            y_predicted_s_indices = y_predicted_s_indices.reshape((-1, 1))
            y_predicted_latlon = convert_spatial_index_array_to_coordinate_array(y_predicted_s_indices, le_grid)
            y_true_latlon = convert_spatial_index_array_to_coordinate_array(y_true, le_grid)
            rmse_meter_mean, error_meter_series = calculate_rmse_on_array(y_predicted_latlon, y_true_latlon)
            logger.info("Model accuracy in timestep %s: %s" % (str(i + 1), rmse_meter_mean))
            X_middle_step = X_middle_step[:, 1:, :]
            y_predicted_s_indices = y_predicted_s_indices.reshape(y_predicted_s_indices.shape[0], 1, y_predicted_s_indices.shape[1])
            X_middle_step = np.concatenate((X_middle_step, y_predicted_s_indices), axis=1)
            y_predicted_latlon = y_predicted_latlon.reshape(y_predicted_latlon.shape[0], 1, y_predicted_latlon.shape[1])
            y_predicted_sequence = np.concatenate((y_predicted_sequence, y_predicted_latlon), axis=1)
    # print(y_predicted_sequence)
    # print(y_predicted_sequence.shape)
    return y_predicted_sequence, error_meter_series


def calculate_rmse_on_array(y_predicted, y_test):
    y_join = np.concatenate((y_predicted, y_test), axis=1)
    # print(y_join)
    # print(y_join.shape)
    error_meter_series = np.apply_along_axis(utility_spatiotemporal_index.calculate_rmse_in_meters, axis=1, arr=y_join)
    # print(error_meter.shape)
    rmse_meter_mean = error_meter_series.mean()
    # logger.info("RMSE in meters: %s" % rmse_meter_mean)
    return rmse_meter_mean, error_meter_series


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
                lat, lon = y[i, -2].item(), (y[i, -1]).item()
                line.append((lon, lat))
            else:
                for j in range(y_length):
                    lat, lon = y[i, j, -2].item(), y[i, j, -1].item()
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
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR

    X_GRID_FILE = s.X_GRID_FILE
    Y_GRID_FILE = s.Y_GRID_FILE


    # X_GRID_FILE = s.X_GRID_EVALUATION_FILE
    # Y_GRID_FILE = s.Y_GRID_EVALUATION_FILE
    # X_MODE_FILE = s.X_MODE_EVALUATION_FILE
    # Y_MODE_FILE = s.Y_MODE_EVALUATION_FILE
    # X_TOPIC_FILE = s.X_TOPIC_EVALUATION_FILE
    # Y_TOPIC_FILE = s.Y_TOPIC_EVALUATION_FILE

    LE_GRID_CLASSES_FILE = s.LE_GRID_CLASSES_FILE

    MODEL_FILE_LSTM_GRID = s.MODEL_FILE_LSTM_GRID
    MODEL_WEIGHT_FILE_LSTM_GRID = s.MODEL_WEIGHT_FILE_LSTM_GRID
    GEOJSON_FILE_OBSERVATION_GRID = s.GEOJSON_FILE_OBSERVATION_GRID
    GEOJSON_FILE_TRUE_GRID = s.GEOJSON_FILE_TRUE_GRID
    GEOJSON_FILE_PREDICTED_LSTM_GRID = s.GEOJSON_FILE_PREDICTED_LSTM_GRID


    le_grid = LabelEncoder()
    le_grid.classes_ = np.load(LE_GRID_CLASSES_FILE)


    X_train, y_train, X_test, y_test = load_grid_dataset(X_GRID_FILE, Y_GRID_FILE, le_grid, EXPERIMENT_PARAMETERS)
    max_s_index = len(le_grid.classes_)
    # np.save(LE_GRID_CLASSES_FILE, le_grid.classes_)
    # print(X_train)

    # Train model
    lstm_model = training_lstm_grid(X_train, y_train, X_test, y_test, EXPERIMENT_PARAMETERS, max_s_index, MODEL_FILE_LSTM_GRID, MODEL_WEIGHT_FILE_LSTM_GRID, FIGURE_DIR)

    # Load model
    lstm_model = load_model(str(MODEL_FILE_LSTM_GRID))

    y_predicted_sequence, error_meter_series = prediction_multiple_steps_lstm_grid(lstm_model, X_test, y_test, le_grid, EXPERIMENT_PARAMETERS)

    X_test_latlon = convert_spatial_index_array_to_coordinate_array(X_test, le_grid, Multistep=True)
    y_test_latlon = convert_spatial_index_array_to_coordinate_array(y_test, le_grid, Multistep=True)

    create_geojson_line_observation(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_OBSERVATION_GRID, EXPERIMENT_PARAMETERS)
    create_geojson_line_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_TRUE_GRID, EXPERIMENT_PARAMETERS)
    create_geojson_line_prediction(X_test_latlon[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], y_predicted_sequence[0:EXPERIMENT_PARAMETERS['VISUALIZATION_SAMPLE_SIZE']], GEOJSON_FILE_PREDICTED_LSTM_GRID, EXPERIMENT_PARAMETERS)

    elapsed = str((datetime.now() - start_time))
    logger.info(f"Finished in: {elapsed}")
    atexit.register(s.exit_handler, __file__, elapsed)

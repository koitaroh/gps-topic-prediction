import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib
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

import settings as s

np.random.seed(7)
random.seed(7)


def define_four_gram(X_train, y_train):
    X_train = X_train[:, -4:-1]
    y_train = y_train[:, 0].reshape(-1, 1, 1)
    # print(X_train.shape)
    # print(y_train.shape)
    gram4_seq = np.concatenate((X_train, y_train), axis=1)
    # print(gram4_seq.shape)
    N = len(gram4_seq)
    dict_gram4 = dict()
    for i in range(N):
        key_gram4 = tuple(gram4_seq[i].ravel())
        if key_gram4 in dict_gram4:
            dict_gram4[key_gram4] += 1
        else:
            dict_gram4[key_gram4] = 1

    gram4 = [(k, dict_gram4[k]) for k in sorted(dict_gram4, key=dict_gram4.get, reverse=True)]
    print('gram4 is created.')
    return gram4


def predict_gram4(gram4, gram3):
    word = next((element[0][3] for element in gram4 if element[0][0:3] == gram3), None)
    if not word:
        word = ''
    return word


def predict(X_test, y_test, gram4):
    N = len(X_test)
    print(f'Testing {N} samples.')
    correct = 0
    unlearned = 0

    for i in range(N):
        gram3 = tuple(X_test[i].ravel())
        prediction = predict_gram4(gram4, gram3)
        if prediction == y_test[i][0][0]:
            correct = + 1
        elif prediction == '':
            unlearned = + 1

    print(f'{correct} correct test.')
    print(f'{unlearned} unlearned test.')

    acc = float(correct) / float(N)
    acc_2 = float(correct) / float(N - unlearned)

    print(f'Accuracy: {acc}.')
    print(f'Accuracy without unlearned: {acc_2}.')


if __name__ == '__main__':
    slack_client = s.sc
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    FIGURE_DIR = s.FIGURE_DIR
    X_GRID_FILE = s.X_GRID_FILE
    Y_GRID_FILE = s.Y_GRID_FILE

    X = np.load(X_GRID_FILE)
    Y = np.load(Y_GRID_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=7)
    gram4 = define_four_gram(X_train, y_train)
    predict(X_test, y_test, gram4)

    # Make notification
    slack_client.api_call("chat.postMessage", channel="#experiment", text=__file__+" is finished.")
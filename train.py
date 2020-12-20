import logging
import os

from datetime import datetime
import autokeras as ak
import numpy as np
import pandas as pd
import json
import csv
import tensorflow as tf
from requests import get

# setup logging
logging.basicConfig(level=logging.DEBUG, format=('%(asctime)s %(levelname)s %(name)s | %(message)s'))
logger = logging.getLogger('darius-trainer')
logger.setLevel(logging.DEBUG)

specs = json.load(open('specs.json', 'r'))

def loadTraindata():
    training_data_json = []
    with open('part2.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            training_data_json.append({'text': row[7], 'spec': row[4]})

        logger.debug(f'{len(training_data_json)} lines of training data loaded')
        training_data_json.pop(0) # headers
        return training_data_json

def train():

    training_data_json = loadTraindata()

    classifier_train_x = []
    classifier_train_y = []

    # build classifier model datastruct
    for row in training_data_json:
        spec = row['spec']
        if spec and spec in specs:
            classifier_train_x.append(row['text'])
            classifier_train_y.append(specs.index(spec))

    logger.debug(f'{len(classifier_train_y)} lines in classifier training data')

    classifier_train_y_hot = tf.one_hot(classifier_train_y, len(specs))
    classifier_train_data = tf.data.Dataset.from_tensor_slices((classifier_train_x, classifier_train_y_hot))

    # train logs
    startTime = datetime.now()

    # tf logs
    log_dir = "/logs/fit/" + datetime.now().strftime("classifier_%m_%d_%Y_%H_%M")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    # classifier model
    classifier = ak.TextClassifier(max_trials=2, overwrite=True, project_name='classifier')
    classifier.fit(classifier_train_data, epochs=40, callbacks=[tensorboard_callback], batch_size=1024)
    logger.debug('training finished')

    # export model to public folder
    classifier_model = classifier.export_model()
    classifier_model.save("/models/clsmdl", save_format="tf")

if __name__ == "__main__":
    # d = loadTraindata()
    # print(d[0])
    # df = pd.read_csv('part2.csv')
    # df.drop(df.columns[[0,1,2,3,5,8,9]], axis=1, inplace=True)
    # print(df.head())
    # print(df['spec'].unique().tolist())
    train()
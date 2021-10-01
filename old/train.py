import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import autokeras as ak
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.saving.save import load_model

HERE = Path(__file__).parent

# setup logging
logging.basicConfig(level=logging.DEBUG, format=('%(asctime)s %(levelname)s %(name)s | %(message)s'))
logger = logging.getLogger('darius-trainer')
logger.setLevel(logging.DEBUG)

specs = json.load(open(HERE.joinpath('specs2.json'), 'r'))

def loadTraindata():
    training_data_json = []
    with open(HERE.joinpath('part2.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            training_data_json.append({'text': row[7], 'spec': row[4]})

        logger.debug(f'{len(training_data_json)} lines of training data loaded')
        training_data_json.pop(0) # headers
        return training_data_json[:15000]

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
    log_dir = "logs/fit/" + datetime.now().strftime("classifier_%m_%d_%Y_%H_%M")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    # classifier model
    classifier = ak.TextClassifier(max_trials=10, overwrite=False, project_name='classifier')
    classifier.fit(classifier_train_data, epochs=8, batch_size=64)
    logger.debug('training finished')

    # export model to public folder
    classifier_model = classifier.export_model()
    classifier_model.save("models/clsmdl", save_format="tf")

def decodeOneHot(encoded):
    """
    decodes the one-hot encoded array, the classifier prediction, to the text representation of the prediction

    Returns the text representation of the provided prediction
    """
    greatestVal = max(encoded)
    greatestIndex = np.where(encoded == greatestVal)[0][0]

    accuracy = str(round(greatestVal*100, 1))
    return specs[greatestIndex], accuracy

def newTrain():
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

    max_features = 20000
    embedding_dim = 128
    sequence_length = 500
    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    
    vectorize_layer.adapt(classifier_train_x)
    v = vectorize_layer(classifier_train_x)

    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(256, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(256, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(len(specs), activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(v, classifier_train_y_hot, epochs=8)

    model.save('new')

    print(model.summary())

def predict():
    max_features = 20000
    embedding_dim = 128
    sequence_length = 500
    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    vectorize_layer.adapt(['r'])
    model = tf.keras.models.load_model('new')

    cls_prediction = decodeOneHot(model.predict(vectorize_layer(['ddddddfollow']))[0])
    prediction = [{'classification': cls_prediction[0], 'confidence': cls_prediction[1]}]

    print(prediction)

if __name__ == "__main__":
    # d = loadTraindata()
    # print(d[0])
    # df = pd.read_csv('part2.csv')
    # df.drop(df.columns[[0,1,2,3,5,8,9]], axis=1, inplace=True)
    # print(df.head())
    # print(df['spec'].unique().tolist())
    train()
    # newTrain()
    predict()

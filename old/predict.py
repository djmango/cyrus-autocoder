import json

import numpy as np
from autokeras import CUSTOM_OBJECTS
from tensorflow.keras.models import load_model

# load models
model = 'train1'
classifier = load_model(f"models/{model}/classifier/best_model", custom_objects=CUSTOM_OBJECTS)

specs = json.load(open('specs.json', 'r'))

def decodeOneHot(encoded):
    """
    decodes the one-hot encoded array, the classifier prediction, to the text representation of the prediction

    Returns the text representation of the provided prediction
    """
    greatestVal = max(encoded)
    greatestIndex = np.where(encoded == greatestVal)[0][0]

    accuracy = str(round(greatestVal*100, 1))
    return specs[greatestIndex], accuracy


def predict(text):
    cls_prediction = decodeOneHot(classifier.predict(np.array([text]))[0])
    prediction = [{'classification': cls_prediction[0], 'confidence': cls_prediction[1]}]
    return prediction

if __name__ == "__main__":
    while True:
        print(predict(input('Descrption of procedings: ')))
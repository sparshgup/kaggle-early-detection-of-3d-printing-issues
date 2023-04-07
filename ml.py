"""
Kaggle: Early Detection of 3D Printing Issues

@author: Sparsh Gupta
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


print("Loading Model")
model = keras.models.load_model("model.h5")


#Loading Test Data

test_data = np.load("testweights.npy")

print("\nTest Data Loaded")

print("Test Data Shape: ", test_data.shape)


#Predictions on Test

test_pred = model.predict(test_data)
test_pred = np.around(test_pred, decimals=0)


# Submission

print("\nSaving predictions on test data to submission.csv")

test_pred = pd.DataFrame(test_pred)

test = pd.read_csv("test.csv")

testpaths = []

for i in range(0, len(test)):
    testpaths.append(test.iloc[i][0])

testpaths = pd.DataFrame(testpaths)

submission = pd.concat([testpaths, test_pred], axis=1)

submission.to_csv("submission.csv", header=["img_path", "has_under_extrusion"], index=False)

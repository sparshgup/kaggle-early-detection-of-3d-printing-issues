"""
Kaggle: Early Detection of 3D Printing Issues

@author: Sparsh Gupta
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Reshape
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.regularizers import l1, l2
from sklearn.metrics import f1_score



#Loading Training Weights

X = np.load("trainweights.npy")

print("\nTrain Data Loaded")

print("Train Data Shape: ", X.shape)

#Loading Labels

y = pd.read_csv("labels.csv")

y = np.array(y)

print("\nLabels Loaded")


#splitting the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

del X
del y

print("\nTraining & Testing Data Shape (for Train Data): ",
      X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#Model

model = Sequential()

model.add(Rescaling(1./255))

model.add(Conv2D(36, 3, activation='relu', input_shape=(240, 320, 3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, 3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))


#Compile Model

optimizer = keras.optimizers.Adam(learning_rate=0.001,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-8)

model.compile(
              loss=keras.losses.BinaryCrossentropy(),
              optimizer=optimizer
              )

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01)


#Fitting the model

history = model.fit(X_train, y_train, epochs=2, batch_size=16,
                    verbose=1, validation_split=0.2, shuffle=True,
                    callbacks=[es_callback])


#Model summary and Save

model.summary()


#Scoring metrics


y_pred = model.predict(X_test)
y_pred = np.around(y_pred, decimals=0)
print(y_pred)

Rt = f1_score(y_test, y_pred)
print("\nF1 Score on test set (train data): ", Rt)

#Save Model

print("Saving Model")
model.save("model.h5")

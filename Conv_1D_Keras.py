
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Convolution1D,MaxPooling1D, Flatten

from keras import backend as K
from sklearn.cross_validation import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import Normalizer
from keras.utils import np_utils

import h5py

X = np.load("mawi_features.pkl")
Y = np.load("mawi_labels.pkl")
C = np.load("mawi_labels.pkl")
T = np.load("mawi_features.pkl")

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

from keras.utils import multi_gpu_model

# Replicates `model` on 5 GPUs.
# This assumes that your machine has 5 available GPUs.
cnn = multi_gpu_model(model, gpus=5)


# In[2]:

cnn = Sequential()
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(15, 1)))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(2, activation="sigmoid"))
print(cnn.summary())
# define optimizer and objective, compile cnn


# In[3]:

cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
# train
cnn.fit(X_train, y_train, epochs=500,validation_data=(X_test, y_test))
# serialize model to JSON
model_json = model.to_json()
with open("cnn1D_MawiLAB.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("cnn1D_MawiLAB.h5")
print("Saved model to disk")


# In[ ]:




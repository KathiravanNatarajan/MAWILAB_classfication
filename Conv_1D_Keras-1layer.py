
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Conv1D,MaxPooling1D, Flatten

from keras.utils import multi_gpu_model
from keras import backend as K
from sklearn.cross_validation import train_test_split

import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import Normalizer
from keras.utils import np_utils

import h5py

X_train = np.load("20170827_mawilab_flow_000features.pkl")
Y_train = np.load("20170827_mawilab_flow_000labels.pkl")

scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)
scaler = Normalizer().fit(Y_train)
Y_train = scaler.transform(Y_train)

# Test Set 1 : 20170827_mawilab_flow_001
X_test1 = np.load("20170827_mawilab_flow_001features.pkl")
Y_test1 = np.load("20170827_mawilab_flow_001labels.pkl")
scaler = Normalizer().fit(X_test1)
X_test1 = scaler.transform(X_test1)
scaler = Normalizer().fit(Y_test1)
Y_test1 = scaler.transform(Y_test1)

# Test Set 2 : 20170827_mawilab_flow_002
X_test2 = np.load("20170827_mawilab_flow_002features.pkl")
Y_test2 = np.load("20170827_mawilab_flow_002labels.pkl")
scaler = Normalizer().fit(X_test2)
X_test2 = scaler.transform(X_test2)
scaler = Normalizer().fit(Y_test2)
Y_test2 = scaler.transform(Y_test2)

# Test Set 3 : 20170827_mawilab_flow_003
X_test3 = np.load("20170827_mawilab_flow_003features.pkl")
Y_test3 = np.load("20170827_mawilab_flow_003labels.pkl")
scaler = Normalizer().fit(X_test3)
X_test3 = scaler.transform(X_test3)
scaler = Normalizer().fit(Y_test3)
Y_test3 = scaler.transform(Y_test3)

# Test Set 4: 20170827_mawilab_flow_004
X_test4 = np.load("20170827_mawilab_flow_004features.pkl")
Y_test4 = np.load("20170827_mawilab_flow_004labels.pkl")
scaler = Normalizer().fit(X_test4)
X_test4 = scaler.transform(X_test4)
scaler = Normalizer().fit(Y_test4)
Y_test4 = scaler.transform(Y_test4)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test1 = np.reshape(X_test1, (X_test1.shape[0],X_test1.shape[1],1))
X_test2 = np.reshape(X_test2, (X_test2.shape[0],X_test2.shape[1],1))
X_test3 = np.reshape(X_test3, (X_test3.shape[0],X_test3.shape[1],1))
X_test4 = np.reshape(X_test4, (X_test4.shape[0],X_test4.shape[1],1))


# In[2]:

cnn1d_1 = Sequential([Conv1D(64, 3, padding="same",input_shape=(16, 1)),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(2),
    Activation('sigmoid'),
])
print(cnn1d_1.summary())


# In[4]:

cnn1d_1 = multi_gpu_model(cnn1d_1, gpus=8)
cnn1d_1.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
# train
start_time = time.time()
cnn1d_1.fit(X_train, Y_train, batch_size=64, validation_data=(X_test2, Y_test2) ,epochs=10)
end_time = time.time() 
print("Total time taken to train the Training model is", (end_time - start_time))
# serialize model to JSON
model_json = cnn1d_1.to_json()
with open("20170827_mawilab_flow_000.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn1d_1.save_weights("20170827_mawilab_flow_000.h5")
print("Saved model to disk")


# In[ ]:




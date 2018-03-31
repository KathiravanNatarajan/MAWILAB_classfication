
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
import matplotlib
matplotlib.use('agg')
import pylab as plt
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import Normalizer
from keras.utils import np_utils
import pickle
import h5py


# KDD - and - 
X_testN_ = pd.read_pickle("./dataset/kdd_test__2labels.pkl").as_matrix()
Y_testN_ = pd.read_pickle("./dataset/kdd_test__2labels_y.pkl").as_matrix()
X_trainN_ = pd.read_pickle("./dataset/kdd_train__2labels.pkl").as_matrix()
Y_trainN_ = pd.read_pickle("./dataset/kdd_train__2labels_y.pkl").as_matrix()
X_trainN_ = np.reshape(X_trainN_, (X_trainN_.shape[0],X_trainN_.shape[1],1))
X_testN_ = np.reshape(X_testN_, (X_testN_.shape[0],X_testN_.shape[1],1))

# KDD + and + 
X_testP_ = pd.read_pickle("./dataset/kdd_test_2labels.pkl").as_matrix()
Y_testP_ = pd.read_pickle("./dataset/kdd_test_2labels_y.pkl").as_matrix()
X_trainP_ = pd.read_pickle("./dataset/kdd_train_2labels.pkl").as_matrix()
Y_trainP_ = pd.read_pickle("./dataset/kdd_train_2labels_y.pkl").as_matrix()
X_trainP_ = np.reshape(X_trainP_, (X_trainP_.shape[0],X_trainP_.shape[1],1))
X_testP_ = np.reshape(X_testP_, (X_testP_.shape[0],X_testP_.shape[1],1))


# KDD - and + 
X_testP_ = pd.read_pickle("./dataset/kdd_test_2labels.pkl").as_matrix()
Y_testP_ = pd.read_pickle("./dataset/kdd_test_2labels_y.pkl").as_matrix()
X_trainN_ = pd.read_pickle("./dataset/kdd_train__2labels.pkl").as_matrix()
Y_trainN_ = pd.read_pickle("./dataset/kdd_train__2labels_y.pkl").as_matrix()
X_trainN_ = np.reshape(X_trainN_, (X_trainN_.shape[0],X_trainN_.shape[1],1))
X_testP_ = np.reshape(X_testP_, (X_testP_.shape[0],X_testP_.shape[1],1))


# KDD + and - 
X_testN_ = pd.read_pickle("./dataset/kdd_test_2labels.pkl").as_matrix()
Y_testN_ = pd.read_pickle("./dataset/kdd_test_2labels_y.pkl").as_matrix()
X_trainP_ = pd.read_pickle("./dataset/kdd_train__2labels.pkl").as_matrix()
Y_trainP_ = pd.read_pickle("./dataset/kdd_train__2labels_y.pkl").as_matrix()
X_trainP_ = np.reshape(X_trainP_, (X_trainP_.shape[0],X_trainP_.shape[1],1))
X_testN_ = np.reshape(X_testN_, (X_testN_.shape[0],X_testN_.shape[1],1))


# In[7]:

cnn1d_1 = Sequential([Conv1D(64, 3, padding="same",input_shape=(124, 1)),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(1),
    Activation('sigmoid'),
])
print(cnn1d_1.summary())


# In[ ]:

#cnn1d_1 = multi_gpu_model(cnn1d_1, gpus=8)
cnn1d_1.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
# train
start_time = time.time()
history = cnn1d_1.fit(X_trainP_, Y_trainP_, batch_size=64, validation_data=(X_testN_, Y_testN_) ,epochs=5)
end_time = time.time() 
print("Total time taken to train the Training model is", (end_time - start_time))
# serialize model to JSON
model_json = cnn1d_1.to_json()
with open("PN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn1d_1.save_weights("PN.h5")
print("Saved model to disk")
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('Acc_ValaccuaracyPN.png')
from keras.utils import plot_model
plot_model(cnn1d_1, to_file='PN.png')


# In[ ]:




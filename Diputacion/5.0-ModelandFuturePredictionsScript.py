#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Masking
from tensorflow.keras.optimizers import Adam
import h5py
import matplotlib.pyplot as plt
import numpy.ma as ma




from matplotlib.pyplot import figure
plt.rcParams["figure.figsize"] = (20,12)
figure(figsize=(100, 80), dpi=80)




tf.random.set_seed(12345)




# pip install -U numpy==1.18.5




physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)




with open('train_X_data.npy', 'rb') as f:
    train_X_data = np.load(f, allow_pickle=True)




with open('train_Y_data.npy', 'rb') as f:
    train_Y_data = np.load(f, allow_pickle=True)




with open('test_X_data.npy', 'rb') as f:
    test_X_data = np.load(f, allow_pickle=True)




with open('test_Y_data.npy', 'rb') as f:
    test_Y_data = np.load(f, allow_pickle=True)




f = h5py.File("diputacion_processed_data.h5","r")
scaled_X = ma.array(f["scaled_x"])
scaled_X.mask = ma.array(f["x_mask"])
minX = np.array(f["minX"])
maxX = np.array(f["maxX"])
f.close()




scaled_X




def inverse_scaler(scaled_x,minX,maxX):
    return scaled_x*maxX-minX




plt.plot(inverse_scaler(scaled_X,minX,maxX))
plt.show()




int(0.25*len(scaled_X))




def partitionSet(test_fraction, data, partition):
    lenX = len(data)
    test_size = int(len(data) * test_fraction)
    test_df = data[int((partition/100)*lenX):int((partition/100)*lenX)+test_size]
    train_df = ma.vstack((data[:int((partition/100)*lenX)-1],data[int((partition/100)*lenX)+test_size:]))
    train_df[int((partition/100)*lenX)] = ma.masked
    return train_df, test_df

train_dataf = []
test_dataf = []

[train_df,test_df] = partitionSet(0.25,scaled_X, 25)




timesteps = 24




X_train = ma.array([train_df[t:t+timesteps] for t in range(0, len(train_df)-timesteps)])
y_train = train_df[timesteps:, :]
X_test = ma.array([test_df[t:t+timesteps] for t in range(0, len(test_df)-timesteps)]) 
y_test = test_df[timesteps:, :]




def LSTM_function(NCells, timesteps, num_features, dropout, NBEpochs, Batchsize, validationSplit):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(timesteps, num_features)))
    model.add(LSTM(1, activation = 'tanh', input_shape = (timesteps, num_features), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(NCells))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer='adam')
    X_ = ma.filled(X_train,0)
    Y_ = ma.filled(y_train,0)
    MODEL = model.fit(X_, Y_, epochs = NBEpochs, batch_size = Batchsize,
                      validation_split = validationSplit, shuffle = False)
    return model, MODEL




X_train.shape[1]




model, MODEL = LSTM_function(64, X_train.shape[1], 1, 0.2, 100, 128, 0.2)




plt.plot(MODEL.history['loss'])
plt.plot(MODEL.history['val_loss'])
plt.show()




prediction_test = model.predict(X_test)




prediction_test.shape




test_pred = ma.masked_invalid(prediction_test)




test_pred[np.where(test_Y_data[1,:,0]==0.0)[0],:] = ma.masked




prediction_train = model.predict(X_train)




train_pred = ma.masked_invalid(prediction_train)




train_pred[[np.where(train_Y_data[1,:,0]==0.0)[0]],:] = ma.masked




plt.plot(y_train[:,0])
plt.plot(train_pred[:,0])
plt.show()




plt.plot(y_test[:,0])
plt.plot(test_pred[:,0])
plt.show()




inputs = scaled_X[len(scaled_X) - len(test_df) - timesteps:]




inputs




inputs = inputs.reshape(-1,1)




X_test = []




for i in range(24, 360):
    X_test.append(inputs[i-timesteps:i,0])




X_test = np.array(X_test)




X_test.shape




X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))




predicted_demand = model.predict(X_test)




plt.plot(inverse_scaler(predicted_demand, minX, maxX))
plt.xlabel('Timesteps')
plt.ylabel('Flow')
plt.savefig('prediction.jpg')







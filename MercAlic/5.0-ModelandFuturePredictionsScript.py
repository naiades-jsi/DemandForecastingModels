#!/usr/bin/env python
# coding: utf-8



# pip install numpy==1.19.2




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




f = h5py.File("mercado_processed_data.h5","r")
scaled_X = ma.array(f["scaled_x"])
scaled_X.mask = ma.array(f["x_mask"])
minX = np.array(f["minX"])
maxX = np.array(f["maxX"])
f.close()




scaled_X




scaled_X.shape




def inverse_scaler(scaled_x,minX,maxX):
    return scaled_x*maxX-minX




plt.plot(inverse_scaler(scaled_X,minX,maxX))
plt.show()




def partitionSet(test_fraction, data, partition):
    lenX = len(data)
    test_size = int(len(data) * test_fraction)
    test_df = data[int((partition/100)*lenX):int((partition/100)*lenX)+test_size]
    train_df = ma.vstack((data[:int((partition/100)*lenX)-1],data[int((partition/100)*lenX)+test_size:]))
    train_df[int((partition/100)*lenX)] = ma.masked
    return train_df, test_df

train_dataf = []
test_dataf = []

[train_df,test_df] = partitionSet(0.25,scaled_X, 8)




timesteps = 48




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
    opt = Adam(learning_rate = 0.00004)
    model.compile(loss = 'mse', optimizer=opt)
    X_ = ma.filled(X_train,0)
    Y_ = ma.filled(y_train,0)
    MODEL = model.fit(X_, Y_, epochs = NBEpochs, batch_size = Batchsize,
                      validation_split = validationSplit, shuffle = False)
    return model, MODEL




model, MODEL = LSTM_function(64, X_train.shape[1], 1, 0.2, 200, 256, 0.2)




plt.plot(MODEL.history['loss'])
plt.plot(MODEL.history['val_loss'])
plt.show()




prediction_test = model.predict(X_test)




prediction_train = model.predict(X_train)




train_pred = ma.masked_invalid(prediction_train)




train_pred[train_pred==train_pred.min()] = ma.masked




plt.plot(y_train[:,0])
plt.plot(train_pred)
plt.show()




plt.plot(y_test[:,0])
plt.plot(prediction_test)
plt.show()




inputs = scaled_X[len(scaled_X) - len(test_df) - timesteps:]




inputs




inputs = inputs.reshape(-1,1)




X_test = []




for i in range(48, 720):
    X_test.append(inputs[i-timesteps:i,0])




X_test = np.array(X_test)




X_test.shape




X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))




predicted_demand = model.predict(X_test)




plt.plot(inverse_scaler(predicted_demand, minX, maxX))
plt.xlabel('Timesteps')
plt.ylabel('Flow')
plt.savefig('prediction.jpg')












#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy.ma as ma
import h5py
from matplotlib.pyplot import figure
plt.rcParams["figure.figsize"] = (20,12)
figure(figsize=(100, 80), dpi=80)




f = h5py.File("benalua_processed_data.h5","r")
scaled_X = ma.array(f["scaled_x"])
scaled_X.mask = ma.array(f["x_mask"])
minX = np.array(f["minX"])
maxX = np.array(f["maxX"])
f.close()




with open('train_performance.npy', 'rb') as f:
    train_performance = np.load(f, allow_pickle=True)




with open('val_performance.npy', 'rb') as f:
    val_performance = np.load(f, allow_pickle=True)




with open('train_X_data.npy', 'rb') as f:
    train_X_data = np.load(f, allow_pickle=True)




with open('train_Y_data.npy', 'rb') as f:
    train_Y_data = np.load(f, allow_pickle=True)




with open('test_X_data.npy', 'rb') as f:
    test_X_data = np.load(f, allow_pickle=True)




with open('test_Y_data.npy', 'rb') as f:
    test_Y_data = np.load(f, allow_pickle=True)




models = []




[models.append(load_model('/home/costa/JoaoModelsForAlicante/Benalua/models/BenaluaModel'+str(i)+'.h5')) for i in range(0,69)]




import os




for i in range(0,len(train_performance)):
    plt.plot(train_performance[i])
    plt.plot(val_performance[i])
    path = "/home/costa/JoaoModelsForAlicante/Benalua/BenaluaPerformance"
    plt.savefig(os.path.join(path, "ModelPerformance"+str(i)+".jpg"))
    plt.show()




def inverse_scaler(data, mindata, maxdata):
    return data*maxdata-mindata




# Prediction on Test Set
def prediction_test(model):
    prediction_test = model.predict(test_X_data[i])
    return prediction_test

test_predictions = []

for i in range(0, len(models)):
    test_predictions.append(prediction_test(models[i]))




# Prediction on Train Set
def prediction_train(model):
    prediction_train = model.predict(train_X_data[i])
    return prediction_train

train_predictions = []

for i in range(0, len(models)):
    train_predictions.append(prediction_train(models[i]))




for i in range(0,len(train_Y_data)):
    plt.plot(inverse_scaler(train_Y_data[i], minX, maxX))
    plt.plot(inverse_scaler(train_predictions[i][:,0], minX, maxX))
    path = "/home/costa/JoaoModelsForAlicante/Benalua/BenaluaTrainPredictions"
    plt.savefig(os.path.join(path, "BenaluaTrainPrediction"+str(i)+".jpg"))
    plt.show()




for i in range(0,len(test_Y_data)):
    plt.plot(inverse_scaler(test_Y_data[i], minX, maxX))
    plt.plot(inverse_scaler(test_predictions[i][:,0], minX, maxX))
    path = "/home/costa/JoaoModelsForAlicante/Benalua/BenaluaTestPredictions"
    plt.savefig(os.path.join(path, "BenaluaTestPrediction"+str(i)+".jpg"))
    plt.show()




def mse(actual, predicted):
    error = np.sum((actual-predicted)**2)/len(actual)
    return error




TrainErrors = []

for i in range(0, len(train_Y_data)-1):
    TrainErrors.append(mse(train_Y_data[i][:,0], train_predictions[i][:,0]))

Sum = 0

for i in range(0, len(TrainErrors)):
    Sum += np.sum(TrainErrors[i])
TrainMeanError = Sum/len(TrainErrors)




TestErrors = []

for i in range(0, len(test_Y_data)-1):
    TestErrors.append(mse(test_Y_data[i][:,0], test_predictions[i][:,0]))
    
for i in range(0, len(TestErrors)):
    Sum += np.sum(TestErrors[i])
TestMeanError = Sum/len(TestErrors)




TrainErrors = np.array(TrainErrors)




TestErrors = np.array(TestErrors)




np.argmin(TrainErrors)




np.argmin(TestErrors)


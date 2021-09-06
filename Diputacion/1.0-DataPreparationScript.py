#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd 
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.rcParams["figure.figsize"] = (20,12)
figure(figsize=(100, 80), dpi=80)




tf.random.set_seed(12345)




df = pd.read_csv('alicante_Diputacion_Flow.csv')




df




df = df.drop_duplicates(subset = 'time', keep='first')




df = df.reset_index(drop=True)




Time = []

for i in range(0, len(df)):
    Time.append(datetime.datetime.fromtimestamp(df['time'][i]).strftime('%Y-%m-%d %H:%M:%S'))




df['time'] = Time




df['value'].isnull().sum()




df.columns = ['Time','Values']




df




plt.plot(df['Values'])




df1 = df.copy()




df1




df15 = df1[:17929].copy()




df15.loc[:,('Time')] = pd.to_datetime(df15['Time'])
df30 = df15.set_index('Time')
df30 = df30.resample('30T').mean()




df30




Time = df30.index
Values= df30['Values']




df = pd.DataFrame({
    'Time' : Time,
    'Values' : Values
})




df.index = [i for i in range(0, len(df))]




df




df['Values'].isnull().sum()




df1[17930:]




df = pd.concat([df, df1[17930:]])




df = df.reset_index()




df.columns




df = df.drop(columns = ['index'])




import plotly.express as px

fig = px.line(df, x='Time', y='Values', title='Flow in Diputacion')
fig.show()




df.to_csv('data_diputacion.csv')




Values = df['Values'].values




Values.shape




import numpy.ma as ma




X = ma.masked_invalid(Values).reshape(-1,1)
X[X==0]=ma.masked




X.min()




from sklearn.preprocessing import MinMaxScaler
# Function for Scaler Application
def Scaler(X):
    #scaler = MinMaxScaler()
    # Fit Scaler
    #scaler_X = scaler.fit(X)
    # Transform Data
    #X_ = scaler_X.transform(X)
    minX = X.min()
    X+=minX
    maxX = X.max()
    X = X/maxX
    return minX,maxX,X




minX,maxX,scaled_x = Scaler(ma.compress_rows(X))




scaled_x.max()




X[-100:]




scaled_x.min()




plt.plot(scaled_x)
plt.show()




scaled_x*maxX-minX




scaled_X = ma.zeros(X.shape)
scaled_X[~X.mask[:,0]] = scaled_x
scaled_X[X.mask] = ma.masked




import h5py




f = h5py.File("diputacion_processed_data.h5","w")
v_ = f.create_dataset("raw_values",shape=Values.shape,dtype="float")
v_[...] = Values

x_ = f.create_dataset("scaled_x",shape=scaled_X.shape,dtype="float")
x_[...] = scaled_X

minx_ = f.create_dataset("minX",shape=(1,),dtype="float")
minx_[...] = minX

maxx_ = f.create_dataset("maxX",shape=(1,),dtype="float")
maxx_[...] = maxX


x_mask = f.create_dataset("x_mask",shape=scaled_X.shape,dtype="Bool")
x_mask[...] = scaled_X.mask

f.close()




f = h5py.File("diputacion_processed_data.h5","r")
scaled_X_ = ma.array(f["scaled_x"])
scaled_X_.mask = ma.array(f["x_mask"])
f.close()




import matplotlib.pyplot as plt




plt.plot(scaled_X_[:500])
plt.plot(scaled_X[:500])
plt.show()

















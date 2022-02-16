from typing import Any, Dict, List
import numpy as np 
import numpy.ma as ma
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Masking
from tensorflow.keras.optimizers import Adam
import numpy.ma as ma

def build_train_model(self, model_structure: Dict[str, Any]):
        def partitionSet(test_fraction, data, partitions):
            lenX = len(data)
            test_size = int(len(data) * test_fraction)
            test_df = data[int((partitions/100)*lenX):int((partitions/100)*lenX)+test_size]
            train_df = ma.vstack((data[:int((partitions/100)*lenX)-1],data[int((partitions/100)*lenX)+test_size:]))
            train_df[int((partitions/100)*lenX)-2] = ma.masked
            return train_df, test_df
        
        def Dataset(train, test, timesteps):
            X_train = ma.array([train[t:t+timesteps] for t in range(0,len(train)-timesteps)])
            y_train = train[timesteps:, :]
            X_test = ma.array([test[t:t+timesteps] for t in range(0,len(test)-timesteps)])
            y_test = test[timesteps:, :]
            return X_train, y_train, X_test, y_test
        
        with tf.device("CPU:0"):
            [self.training_dataf, self.testing_dataf] = partitionSet(model_structure["test_size"], self.training_data, 100-model_structure["test_size"]*100)
            [self.training_X_data, self.training_Y_data, self.testing_X_data, self.testing_Y_data] = Dataset(self.training_dataf, self.testing_dataf, 24)
            self.nn = Sequential()
            self.nn.add(Masking(mask_value=0., input_shape=(model_structure["n_of_timesteps"], model_structure["num_features"])))
            self.nn.add(LSTM(1, activation = 'tanh', input_shape = (model_structure["n_of_timesteps"], model_structure["num_features"]), return_sequences=True))
            self.nn.add(Dropout(model_structure["dropout"]))
            self.nn.add(LSTM(model_structure["n_of_neurons"]))
            self.nn.add(Dropout(model_structure["dropout"]))
            self.nn.add(Dense(1))
            self.nn.compile(loss = 'mse', optimizer='adam')

            X_ = ma.filled(self.training_X_data,0)
            Y_ = ma.filled(self.training_Y_data,0)

            self.model = self.nn.fit(X_, Y_, epochs = model_structure["epochs"], batch_size = model_structure["batch_size"],
                            validation_split = model_structure["validation_split"], shuffle = False, verbose = 0)
            self.save_model(self.model_name)

def save_model(self, filename):
    self.nn.save("models/" + filename + "_LSTM")
    #print("Saving GAN")

def load_model(self, filename):
    self.Model = tf.keras.models.load_model(filename + "_LSTM")

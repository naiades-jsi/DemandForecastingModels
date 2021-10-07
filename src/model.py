from typing import Any, Dict, List
import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Masking
from tensorflow.keras.optimizers import Adam
import h5py
import numpy.ma as ma

# Train on all the data 

class Model():
    training_X_data: str
    testing_X_data: str
    training_Y_data: str
    testing_Y_data: str
    model_structure: Dict[str, Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        self.training_X_data = conf["training_X_data"]
        self.testing_X_data = conf["testing_Y_data"]
        self.training_Y_data = conf["training_X_data"]
        self.testing_Y_data = conf["testing_Y_data"]
        self.model_structure = conf["model_structure"]

        # Build and train the model
        self.build_train_model(model_structure=self.model_structure,
                               train_X_file=self.training_X_data,
                               test_X_file = self.testing_X_data,
                               train_Y_file = self.training_Y_data,
                               test_Y_data = self.testing_Y_data)

    # def inverse_scaler(data, mindata, maxdata):
    #     return data*maxdata-mindata

    def mse(actual, predicted):
        error = np.sum((actual-predicted)**2)/len(actual)
        return error

    def message_insert(self, message_value: Dict[str, Any]) -> Any:
        # Open .h5 file with processed data

        f = h5py.File("alipark_processed_data.h5","r")
        scaled_X = ma.array(f["scaled_x"])
        scaled_X.mask = ma.array(f["x_mask"])
        minX = np.array(f["minX"])
        maxX = np.array(f["maxX"])
        f.close()

        # Partition dataset into training and testing sets

        def partitionSet(test_fraction, data, partition):
            lenX = len(data)
            test_size = int(len(data) * test_fraction)
            test_df = data[int((partition/100)*lenX):int((partition/100)*lenX)+test_size]
            train_df = ma.vstack((data[:int((partition/100)*lenX)-1],data[int((partition/100)*lenX)+test_size:]))
            train_df[int((partition/100)*lenX)] = ma.masked
            return train_df, test_df

        timesteps = 24

        # train_dataf = []
        # test_dataf = []

        [train_df,test_df] = partitionSet(0.25,scaled_X, 54)
        
        model = load_model('/home/costa/JoaoModelsForAlicante/AliPark/models/AliParkModel54.h5')
        prediction_train = model.predict(self.training_X_data[54])
        prediction_test = model.predict(self.testing_X_data[54])
        inputs = scaled_X[len(scaled_X) - len(test_df) - timesteps:]
        inputs = inputs.reshape(-1,1)
        X_test = []
        for i in range(24, 360):
            X_test.append(inputs[i-timesteps:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_demand = model.predict(X_test)

    def build_train_model(self, model_structure: Dict[str, Any], train_X_file: str, training_Y_file: str):
        # for i in range(0, len(self.training_data)):
            self.nn = Sequential()
            self.nn.add(Masking(mask_value=0., input_shape=(model_structure["n_of_timesteps"], model_structure["num_features"])))
            self.nn.add(LSTM(1, activation = 'tanh', input_shape = (model_structure["n_of_timesteps"], model_structure["num_features"]), return_sequences=True))
            self.nn.add(Dropout(model_structure["dropout"]))
            self.nn.add(LSTM(model_structure["n_of_neurons"]))
            self.nn.add(Dropout(model_structure["dropout"]))
            self.nn.add(Dense(1))
            self.nn.compile(loss = 'mse', optimizer='adam')
            X_ = ma.filled(self.training_X_data[54],0)
            Y_ = ma.filled(self.training_Y_data[54],0)
            MODEL = self.nn.fit(X_, Y_, epochs = model_structure["epochs"], batch_size = model_structure["batch_size"],
                            validation_split = model_structure["validation_split"], shuffle = False)
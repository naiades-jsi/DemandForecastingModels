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
import time

class Model():
    training_X_data: str
    model_structure: Dict[str, Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        self.training_X_data = conf["training_X_data"]
        self.training_Y_data = conf["training_Y_data"]
        self.model_structure = conf["model_structure"]
        self.horizon = conf["horizon"]

        # Build and train the model
        self.build_train_model(model_structure=self.model_structure,    
                               train_X_file=self.training_X_data,
                               train_Y_file = self.training_Y_data)

    def message_insert(self, message_value: Dict[str, Any]) -> Any:
        ftr_vector = message_value['ftr_vector']
        ftr_vector = np.array(ftr_vector)
        predicted_demand = self.model.predict(ftr_vector)
        output_dictionary = {"timestamp": message_value['timestamp'], 
        "timestamp_prediction": time.time(), 
        "value": predicted_demand, 
        "horizon": self.horizon}

    def build_train_model(self, model_structure: Dict[str, Any]):
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
                        validation_split = model_structure["validation_split"], shuffle = False)
        self.model.save('/home/costa/JoaoModelsForAlicante/Autmeasurements_node_alicante_autobuses_flowoBus/models/AutobusModel.h5')
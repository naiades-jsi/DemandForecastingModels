from typing import Any, Dict, List
import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Masking
from tensorflow.keras.optimizers import Adam

from output import OutputAbstract, KafkaOutput

import h5py
import numpy.ma as ma
import time

class LSTM_model():
    training_X_data: str
    model_structure: Dict[str, Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:

        filename_X = 'training_data/' + conf["training_X_data"] + '.npy'
        filename_Y = 'training_data/' + conf["training_Y_data"] + '.npy'
        self.training_X_data = np.load(filename_X, allow_pickle= True)
        self.training_Y_data = np.load(filename_Y, allow_pickle=True)
        self.model_structure = conf["model_structure"]
        self.horizon = conf["horizon"]
        self.model_name = conf["model_name"]

        # OUTPUT/VISUALIZATION INITIALIZATION & CONFIGURATION
        self.outputs = [eval(o) for o in conf["output"]]
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

        # Build and train the model
        self.build_train_model(model_structure=self.model_structure)

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        ftr_vector = message_value['ftr_vector']
        ftr_vector = np.array(ftr_vector)
        print("ftr_vector:" + str(ftr_vector))

        timestamp = message_value["timestamp"]

        predicted_demand = [float(k) for k in self.nn.predict(np.atleast_2d(ftr_vector))[0]]
        output_dictionary = {"timestamp": message_value['timestamp'], 
        "value": predicted_demand, 
        "horizon": self.horizon,
        "prediction_time": time.time()}

        for output in self.outputs:
            output.send_out(timestamp=timestamp,
                            value=output_dictionary,
                            suggested_value = None, 
                            algorithm= 'LSTM model')

    def build_train_model(self, model_structure: Dict[str, Any]):
        self.nn = Sequential()
        self.nn.add(Masking(mask_value=0., input_shape=(model_structure["n_of_timesteps"], model_structure["num_features"])))
        self.nn.add(LSTM(1, activation = 'tanh', input_shape = (model_structure["n_of_timesteps"], model_structure["num_features"]), return_sequences=True))
        self.nn.add(Dropout(model_structure["dropout"]))
        self.nn.add(LSTM(model_structure["n_of_neurons"]))
        self.nn.add(Dropout(model_structure["dropout"]))
        self.nn.add(Dense(1))
        self.nn.compile(loss = 'mse', optimizer='adam')

        X_ = ma.filled(self.training_X_data[0],0)
        Y_ = ma.filled(self.training_Y_data[0],0)

        self.model = self.nn.fit(X_, Y_, epochs = model_structure["epochs"], batch_size = model_structure["batch_size"],
                        validation_split = model_structure["validation_split"], shuffle = False, verbose = 0)
        #self.model.save('/home/costa/JoaoModelsForAlicante/Autmeasurements_node_alicante_autobuses_flowoBus/models/AutobusModel.h5')
        self.save_model(self.model_name)

    def save_model(self, filename):
        self.nn.save("models/" + filename + "_LSTM")
        #print("Saving GAN")

    def load_model(self, filename):
        self.model = keras.models.load_model(filename + "_LSTM")
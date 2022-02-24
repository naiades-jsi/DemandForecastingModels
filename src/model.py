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
from src.output import OutputAbstract, KafkaOutput
import h5py
import numpy.ma as ma
import time
import matplotlib.pyplot as plt

class LSTM_model():
    training_data: str
    model_structure: Dict[str, Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)
        self.feature_vector_array = []

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:

        #data = pd.read_csv(conf['training_data'])
        #Values = data['Values'].values
        #X = ma.masked_invalid(Values).reshape(-1,1)
        #X[X==0] = ma.masked
        #def Scaler(X):
        #    minX = X.min()
        #    maxX = X.max()
        #    X-=minX
        #    X = X/(maxX-minX)
        #    return minX,maxX,X
        #minX,maxX,scaled_x = Scaler(ma.compress_rows(X))    
        #self.training_data = scaled_x
        #self.model_structure = conf["model_structure"]
        #self.horizon = conf["horizon"]

        # Saved model file
        self.model_file=conf["model_file"]

        # Model name
        self.model_name = conf["model_name"]

        # Data File to extract min, max
        self.data = conf['data']
        
        # timesteps in the futurte to predict
        self.predicted_timesteps = conf["predicted_timesteps"]

        if ("model_file" in conf):
            self.model_file = conf["model_file"]
            self.load_model(self.model_file)

        # OUTPUT/VISUALIZATION INITIALIZATION & CONFIGURATION
        self.outputs = [eval(o) for o in conf["output"]]
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

        # TODO: add training data location to the configuration

    def min_max_of_data(self, file_location):
        # TODO save min and max
        data = pd.read_csv(file_location)
        values = np.array(data['Values'])
        min = values.min()
        max = values.max()
        return min, max

    def load_model(self, filename):
        load_model(filename)

        # Load model
        self.nn = self.load_model(self.model_file)

    #def save_model(self, filename):
    #    self.nn.save("LoadedModels/" + filename + "_LSTM")
        #print("Saving GAN")

    def feature_vector_creation(self, message_value: Dict[Any, Any]) -> Any:
        print(message_value)
        value = message_value["ftr_vector"]
        timestamp = message_value["timestamp"]

        self.feature_vector_array.append(value)

        if(len(self.feature_vector_array[-1]) != 24):
            print("not enough values")
            return

        dict_to_insert = {
            "ftr_vector": self.feature_vector_array,
            "timestamp": timestamp
        }

        self.message_insert(message_value=dict_to_insert)

        self.feature_vector_array = self.feature_vector_array[1:]

    def feature_vector_normalization(self, ftr_vector):
        minX, maxX = self.min_max_of_data(self.data)
        scaled_ftr_vector = (ftr_vector-minX)/(maxX-minX)
        return scaled_ftr_vector

    def reverse_normalization(self, predictions):
        minX, maxX = self.min_max_of_data(self.data)
        reverse_predictions = predictions*(maxX-minX)+minX
        return reverse_predictions

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        ftr_vector = message_value['ftr_vector']
        timestamp = message_value["timestamp"]
        ftr_vector = np.array(ftr_vector)
        print(ftr_vector)
        predictions = []
        scaled_ftr_vector = self.feature_vector_normalization(ftr_vector)
        print(scaled_ftr_vector)
        for i in range(self.predicted_timesteps):
            predicted_demand = np.array([float(k) for k in self.nn.predict(np.atleast_2d(scaled_ftr_vector))])
            predictions.append(predicted_demand)
            scaled_ftr_vector.insert(0, predicted_demand)
            scaled_ftr_vector = scaled_ftr_vector[:-1]
            #scaled_ftr_vector-=minX
            #scaled_ftr_vector = ftr_vector/(maxX-minX)
        #print("scaled_ftr_vector" + str(scaled_ftr_vector))
        
        predicted_results = self.reverse_normalization(predictions)
        #predicted_results = [predicted_demand[i]*(maxX-minX)+minX for i in range(0, len(predicted_demand))]

        #{"1h": 0,
        # "2h": 1}
        
        output_dictionary = {"timestamp": message_value['timestamp'],
        "value": predicted_results,
        #"horizon": self.horizon,
        "prediction_time": time.time()}

        for output in self.outputs:
            output.send_out(timestamp=timestamp,
                            value=output_dictionary,
                            suggested_value = None, 
                            algorithm= 'LSTM model')
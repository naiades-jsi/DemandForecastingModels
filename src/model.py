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
        self.model_file=conf["model_file"]
        #self.horizon = conf["horizon"]
        self.model_name = conf["model_name"]

        if ("model_file" in conf):
            self.model_file = conf["model_file"]

        # OUTPUT/VISUALIZATION INITIALIZATION & CONFIGURATION
        self.outputs = [eval(o) for o in conf["output"]]
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(output_configurations[o])

        # Build and train the model
        #self.build_train_model(model_structure=self.model_structure)
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

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        self.nn = load_model(self.model_file)
        ftr_vector = message_value['ftr_vector']
        ftr_vector = np.array(ftr_vector)
        print("ftr_vector:" + str(ftr_vector))
        minX = ftr_vector.min()
        maxX = ftr_vector.max()
        ftr_vector-=minX
        scaled_ftr_vector = ftr_vector/(maxX-minX)
        print("scaled_ftr_vector" + str(scaled_ftr_vector))
        timestamp = message_value["timestamp"]
        predicted_demand = np.array([float(k) for k in self.nn.predict(np.atleast_2d(scaled_ftr_vector))])
        predicted_results = []
        predicted_results = [predicted_demand[i]*(maxX-minX)+minX for i in range(0, len(predicted_demand))]
        output_dictionary = {"timestamp": message_value['timestamp'],
        "value": predicted_results,
        #"horizon": self.horizon,
        "prediction_time": time.time()}
        for i in range(0, len(predicted_demand)):
            plt.plot(np.concatenate([ftr_vector, predicted_demand[i]]))

        for output in self.outputs:
            output.send_out(timestamp=timestamp,
                            value=output_dictionary,
                            suggested_value = None, 
                            algorithm= 'LSTM model')
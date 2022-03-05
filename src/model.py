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


physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
print(physical_devices)

class LSTM_model():
    training_data: str
    model_structure: Dict[str, Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)
        self.feature_vector_array = []
        self.scaled_feature_vector_array = []

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:

        # Saved model file
        self.model_file=conf["model_file"]

        # Model name
        self.model_name = conf["model_name"]

        # Data File to extract min, max
        self.data = conf["data"]
        
        # timesteps in the futurte to predict
        self.predicted_timesteps = conf["predicted_timesteps"]
    
        # number of days to predict
        self.n_days = conf["n_days"]

        self.n_features = conf["n_features"]

        if ("model_file" in conf):
            print("loading_model")
            self.model_file = conf["model_file"]
            self.model = self.load_model(self.model_file)

        self.outputs = [eval(o) for o in conf["output"]]
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(conf = output_configurations[o])

    def min_max_of_data(self, file_location):
        data = pd.read_csv(file_location)
        min = data['Values'].min()
        max = data['Values'].max()
        return min, max

    def load_model(self, filename):
        return(load_model(filename))

    def feature_vector_creation(self, message_value: Dict[Any, Any]) -> Any:
        scaled_value = self.feature_vector_normalization(message_value["ftr_vector"])
        value = message_value["ftr_vector"]
        timestamp = message_value["timestamp"]

        if(len(scaled_value) != self.predicted_timesteps*self.n_features):
            return
        else:
            self.scaled_feature_vector_array.append(scaled_value)
            self.feature_vector_array.append(value)

            #if(len(self.feature_vector_array[-1]) != 24):
            #    print("not enough values")
            #    return

            #print("\n")
            dict_to_insert = {
                "scaled_ftr_vector": self.scaled_feature_vector_array,
                "ftr_vector": self.feature_vector_array,
                "timestamp": timestamp
            }

            if(len(self.scaled_feature_vector_array) > self.n_days*self.predicted_timesteps):
                self.scaled_feature_vector_array = self.scaled_feature_vector_array[-self.n_days*self.predicted_timesteps:]
                self.feature_vector_array = self.feature_vector_array[-self.n_days*self.predicted_timesteps:]

            self.message_insert(message_value=dict_to_insert)

    def feature_vector_normalization(self, ftr_vector):
        [minX, maxX] = self.min_max_of_data(self.data)
        scaled_ftr_vector = (np.array(ftr_vector)-minX)/(maxX-minX)
        return scaled_ftr_vector

    def reverse_normalization(self, predictions):
        [minX, maxX] = self.min_max_of_data(self.data)
        reverse_predictions = np.array(predictions)*(maxX-minX)+minX
        return reverse_predictions

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        
        #timesteps = 24
        #data = pd.read_csv(self.data)
        #values = data['Values']
        #test_component = values[int(0.8*len(values))+1:]
        #ftr_vector = np.array([test_component[t:t+timesteps] for t in range(0, len(test_component)-timesteps)])
        ftr_vector = np.array(message_value['ftr_vector'])
        scaled_ftr_vector = self.feature_vector_normalization(ftr_vector)
        
        timestamp = message_value["timestamp"]
        n_future = self.n_days*self.predicted_timesteps

        if(scaled_ftr_vector.shape[0] == n_future):


            scaled_forecast = self.model.predict(scaled_ftr_vector.reshape(scaled_ftr_vector.shape[0], self.predicted_timesteps, self.n_features))

            # To inverse scale it
            predictions = self.reverse_normalization(scaled_forecast)
        else:
            predictions = None


        # where X_train.shape() is (n_future, timesteps, 1)
        #plt.plot(ftr_vector[0,0,:][-n_future:])
        #plt.show()
        #plt.plot(predictions[:,0])
        #plt.show()

        if(predictions is not None):
            for output in self.outputs:
                # Create output dictionary
                output_dictionary = {
                    "timestamp": message_value['timestamp'],
                    "value": str(list(predictions.flatten())),
                    "prediction_time": time.time()}
                
                # Send out
                output.send_out(timestamp=timestamp,
                                value=output_dictionary,
                                suggested_value = None, 
                                algorithm= 'LSTM model')


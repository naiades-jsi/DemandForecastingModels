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
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
print(physical_devices)

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
        self.data = conf["data"]
        
        # timesteps in the futurte to predict
        self.predicted_timesteps = conf["predicted_timesteps"]

        if ("model_file" in conf):
            self.model_file = conf["model_file"]
            self.load_model(self.model_file)

        # OUTPUT CONFIGURATION
        output_configs = [eval(o) for o in conf["output"]]
        outputs = []
        for o in output_configs:
            output = KafkaOutput().configure(conf={"output_topic": o["topic"]})
            self.outputs.append([o["mask"], output, o["horizon"]])

    def min_max_of_data(self, file_location):
        data = pd.read_csv(file_location)
        min = data['Values'].min()
        max = data['Values'].max()
        return min, max

    def load_model(self, filename):
        load_model(filename)

    def feature_vector_creation(self, message_value: Dict[Any, Any]) -> Any:
        print(message_value)
        value = message_value["ftr_vector"]
        timestamp = message_value["timestamp"]

        self.feature_vector_array.append(value)

        if(len(self.feature_vector_array[-1]) != 24):
            print("not enough values")
            return

        #print("\n")
        dict_to_insert = {
            "ftr_vector": self.feature_vector_array,
            "timestamp": timestamp
        }


        self.message_insert(message_value=dict_to_insert)

        self.feature_vector_array = self.feature_vector_array[1:]

    def feature_vector_normalization(self, ftr_vector):
        [minX, maxX] = self.min_max_of_data(self.data)
        scaled_ftr_vector = (ftr_vector-minX)/(maxX-minX)
        return scaled_ftr_vector

    def reverse_normalization(self, predictions):
        [minX, maxX] = self.min_max_of_data(self.data)
        reverse_predictions = np.array(predictions, dtype="object")*(maxX-minX)+minX
        return reverse_predictions

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        model = load_model(self.model_file)
        ftr_vector = message_value['ftr_vector']
        timestamp = message_value["timestamp"]
        ftr_vector = np.array(ftr_vector)
        ftr_vector = ftr_vector[0,:]
        predictions = []
        scaled_ftr_vector = self.feature_vector_normalization(ftr_vector)
        for i in range(self.predicted_timesteps):
            predicted_demand = np.array([float(k) for k in model.predict(np.atleast_2d(scaled_ftr_vector))])
            predictions.append(predicted_demand)
            scaled_ftr_vector = np.hstack((predictions[i], scaled_ftr_vector[:-1]))
        
        predicted_results = self.reverse_normalization(predictions).tolist()
        prediction_time = time.time()

        # Output
        for output_tuple in self.outputs:
            # Get the mask and select the correct prediction
            mask = output_tuple[0]
            output_dictionary = {"timestamp": timestamp,
                "value": predictions[mask],
                "horizon": output_tuple[2],
                "prediction_time": prediction_time}

            # Get the output object and send the message out
            output = output_tuple[1]
            output.send_out(timestamp=timestamp,
                            value=output_dictionary,
                            suggested_value = None, 
                            algorithm= 'LSTM model')

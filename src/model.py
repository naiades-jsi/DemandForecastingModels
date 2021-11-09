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
from output import OutputAbstract, KafkaOutput
import h5py
import numpy.ma as ma
import time

class LSTM_model():
    training_data: str
    model_structure: Dict[str, Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:

        data = pd.read_csv(conf['training_data'])
        Values = data['Values'].values
        X = ma.masked_invalid(Values).reshape(-1,1)
        X[X==0] = ma.masked
        def Scaler(X):
            minX = X.min()
            X+=minX
            maxX = X.max()
            X = X/maxX
            return minX,maxX,X
        minX,maxX,scaled_x = Scaler(ma.compress_rows(X))    
        self.training_data = scaled_x
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
        X_ = tf.convert_to_tensor(X_)
        Y_ = tf.convert_to_tensor(Y_)

        self.model = self.nn.fit(X_, Y_, epochs = model_structure["epochs"], batch_size = model_structure["batch_size"],
                        validation_split = model_structure["validation_split"], shuffle = False, verbose = 0)
        self.save_model(self.model_name)

    def save_model(self, filename):
        self.nn.save("models/" + filename + "_LSTM")
        #print("Saving GAN")

    def load_model(self, filename):
        self.Model = tf.keras.models.load_model(filename + "_LSTM")

from typing import Any, Dict, List
import numpy as np
import numpy.ma as ma
import pandas as pd
#import tensorflow as tf
import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Masking
#from tensorflow.keras.optimizers import Adam
from src.output import OutputAbstract, KafkaOutput
import h5py
import numpy.ma as ma
import time
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import logging

# logging
LOGGER = logging.getLogger("wf-monitor")
logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO)

# not loading TF for Alicante (uncomment for Braila)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# print(physical_devices)

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

        #number of features used
        self.n_features = conf["n_features"]

        #model loading
        if ("model_file" in conf):
            print("loading_model", flush = True)
            self.model_file = conf["model_file"]
            self.model = self.load_model(self.model_file)

        #missing data imputer configuration
        if ("fill_missing_data" in conf):
            print("loading_data", flush = True)
            self.max_missing_data_memory = conf["max_missing_data_memory"]
            self.missing_data_memory = np.load(conf["fill_missing_data"], allow_pickle=True)
            self.imputer = KNNImputer(n_neighbors=4)

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
        value = message_value["ftr_vector"]
        timestamp = message_value["timestamp"]

        value = self.fill_missing_data(np.array([value]))

        scaled_value = self.feature_vector_normalization(value)


        if(len(scaled_value) != int(self.predicted_timesteps/2*self.n_features)):
            return
        else:
            self.scaled_feature_vector_array.append(scaled_value)
            self.feature_vector_array.append(value)

            #print(len(self.feature_vector_array))

            if(len(self.scaled_feature_vector_array) > self.n_days*self.predicted_timesteps):
                self.scaled_feature_vector_array = self.scaled_feature_vector_array[-self.n_days*self.predicted_timesteps:]
                self.feature_vector_array = self.feature_vector_array[-self.n_days*self.predicted_timesteps:]

            dict_to_insert = {
                "scaled_ftr_vector": self.scaled_feature_vector_array,
                "ftr_vector": self.feature_vector_array,
                "timestamp": timestamp
            }

            self.message_insert(message_value=dict_to_insert)

    def feature_vector_normalization(self, ftr_vector):
        [minX, maxX] = self.min_max_of_data(self.data)
        scaled_ftr_vector = (np.array(ftr_vector)-minX)/(maxX-minX)
        return scaled_ftr_vector

    def reverse_normalization(self, predictions):
        [minX, maxX] = self.min_max_of_data(self.data)
        reverse_predictions = np.array(predictions)*(maxX-minX)+minX
        return reverse_predictions

    def fill_missing_data(self, ftr_vector):

        #Append incoming FV to missing data memory
        self.missing_data_memory = np.concatenate([self.missing_data_memory, ftr_vector])

        if(len(self.missing_data_memory) > self.max_missing_data_memory):
            self.missing_data_memory = self.missing_data_memory[-self.max_missing_data_memory:]

        #Fill missing data. Last value is the incoming FV.
        filled_ftr_vector = self.imputer.fit_transform(self.missing_data_memory)[-1]

        return filled_ftr_vector

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:

        ftr_vector = np.array(message_value['ftr_vector'])
        #ftr_vector = self.fill_missing_data(ftr_vector)

        scaled_ftr_vector = self.feature_vector_normalization(ftr_vector)
        #print(f'input FV shape: {scaled_ftr_vector.shape}')
        timestamp = message_value["timestamp"]
        n_future = self.n_days*self.predicted_timesteps

        #print(f'FV: {scaled_ftr_vector}')

        try:

            scaled_forecast = self.model.predict(scaled_ftr_vector.reshape((scaled_ftr_vector.shape[0], int(self.predicted_timesteps/2), self.n_features), order = 'C'))

            # To inverse scale it
            predictions = self.reverse_normalization(scaled_forecast)
        except:
            print('Exception in LSTM prediction.', flush = True)
            predictions = None


        # where X_train.shape() is (n_future, timesteps, 1)
        #plt.plot(ftr_vector[0,0,:][-n_future:])
        #plt.show()
        #plt.plot(predictions[:,0])
        #plt.show()

        if(predictions is not None):
            for output in self.outputs:

                if(len(predictions.flatten()) < n_future):
                    out_array = np.concatenate([predictions.flatten(), np.zeros(n_future - len(predictions.flatten()))])
                else:
                    out_array = predictions.flatten()
                # Create output dictionary
                output_dictionary = {
                    "timestamp": message_value['timestamp'],
                    "value": list([float(x) for x in out_array]),
                    "prediction_time": time.time()}

                #print(f'Predictions: {predictions}')
                #print(f'Output: {output_dictionary}')

                # Send out
                output.send_out(timestamp=timestamp,
                                value=output_dictionary,
                                suggested_value = None,
                                algorithm= 'LSTM model')



class GDB_model():
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

        #number of features used
        self.n_features = conf["n_features"]

        #model loading
        if ("model_file" in conf):
            LOGGER.info("Loading_model: %s", conf["model_file"])
            self.model_file = conf["model_file"]
            self.model = self.load_model(self.model_file)

        # missing data imputer configuration
        if ("fill_missing_data" in conf):
            LOGGER.info("Loading missing data %s", conf["fill_missing_data"])
            self.max_missing_data_memory = conf["max_missing_data_memory"]
            self.missing_data_memory = np.load(conf["fill_missing_data"], allow_pickle=True)
            self.imputer = KNNImputer(n_neighbors=4)

        self.outputs = [eval(o) for o in conf["output"]]
        output_configurations = conf["output_conf"]
        for o in range(len(self.outputs)):
            self.outputs[o].configure(conf = output_configurations[o])

    def load_model(self, filename):
        """
        Load a model from a file

        Parameters
        ----------
        filename : str
            The name of the file to load the model from

        Returns
        -------
        model : object
            The loaded model
        """

        loaded_model = pickle.load(open(filename, 'rb'))
        return(loaded_model)

    def feature_vector_creation(self, message_value: Dict[Any, Any]) -> Any:
        """
        Create feature vector from incoming message. Also takes care of the
        missing data imputation and adds time of day and day of week features.

        Parameters
        ----------
        message_value : dict
            The incoming message

        Returns
        -------
        ftr_vector : np.array
            The feature vector
        """

        value = message_value["ftr_vector"]
        timestamp = message_value["timestamp"] / 1000

        value = self.fill_missing_data(np.array([value]))

        # add hour and day data to the feature vector

        # create date-time from unix timestamp
        date_time = datetime.datetime.fromtimestamp(timestamp)

        # add hour and day data to the feature vector
        value = np.append(value, date_time.hour)
        value = np.append(value, date_time.weekday())

        # we do not need scaling for GDB
        scaled_value = value

        if(len(scaled_value) != (int(self.predicted_timesteps/2*self.n_features)) + 2):
            LOGGER.error("Feature vector length does not match the model input length")
            return
        else:
            LOGGER.info("Feature vector OK - prediction phase started, %s", self.model_name)

            self.scaled_feature_vector_array.append(scaled_value)
            self.feature_vector_array.append(value)

            #print(len(self.feature_vector_array))

            if(len(self.scaled_feature_vector_array) > self.n_days*self.predicted_timesteps):
                self.scaled_feature_vector_array = self.scaled_feature_vector_array[-self.n_days*self.predicted_timesteps:]
                self.feature_vector_array = self.feature_vector_array[-self.n_days*self.predicted_timesteps:]

            dict_to_insert = {
                "scaled_ftr_vector": self.scaled_feature_vector_array,
                "ftr_vector": self.feature_vector_array,
                "timestamp": timestamp
            }

            self.message_insert(message_value=dict_to_insert)

    def fill_missing_data(self, ftr_vector):
        """
        Fill missing data in the feature vector using KNN imputer.

        Parameters
        ----------
        ftr_vector : np.array
            The feature vector

        Returns
        -------
        ftr_vector : np.array
            The feature vector with missing data filled
        """
        # append incoming FV to missing data memory
        self.missing_data_memory = np.concatenate([self.missing_data_memory, ftr_vector])

        if(len(self.missing_data_memory) > self.max_missing_data_memory):
            self.missing_data_memory = self.missing_data_memory[-self.max_missing_data_memory:]

        # fill missing data. Last value is the incoming FV.
        filled_ftr_vector = self.imputer.fit_transform(self.missing_data_memory)[-1]

        return filled_ftr_vector

    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        """
        Create prediction for the incoming message and send it out.

        Parameters
        ----------
        message_value : dict
            The incoming message

        Returns
        -------
        None
        """

        ftr_vector = np.array(message_value['ftr_vector'])
        #ftr_vector = self.fill_missing_data(ftr_vector)

        # scaled_ftr_vector = self.feature_vector_normalization(ftr_vector)
        scaled_ftr_vector = ftr_vector
        #print(f'input FV shape: {scaled_ftr_vector.shape}')
        timestamp = message_value["timestamp"]
        n_future = self.n_days*self.predicted_timesteps

        #print(f'FV: {scaled_ftr_vector}')

        try:
            LOGGER.info("Predicting: %s", self.model_name)
            scaled_forecast = self.model.predict(scaled_ftr_vector.reshape((scaled_ftr_vector.shape[0], int(self.predicted_timesteps/2) + 2, self.n_features), order = 'C'))

            # To inverse scale it
            # predictions = self.reverse_normalization(scaled_forecast)
            predictions = scaled_forecast
        except Exception as e:
            LOGGER.error('Exception in GDB prediction: %s', self.model_name)
            LOGGER.exception(str(e))
            predictions = None


        # where X_train.shape() is (n_future, timesteps, 1)
        #plt.plot(ftr_vector[0,0,:][-n_future:])
        #plt.show()
        #plt.plot(predictions[:,0])
        #plt.show()

        if (predictions is not None):
            for output in self.outputs:

                if(len(predictions.flatten()) < n_future):
                    out_array = np.concatenate([predictions.flatten(), np.zeros(n_future - len(predictions.flatten()))])
                else:
                    out_array = predictions.flatten()

                # Create output dictionary
                output_dictionary = {
                    "timestamp": message_value['timestamp'],
                    "value": list([float(x) for x in out_array]),
                    "prediction_time": time.time()}

                #print(f'Predictions: {predictions}')
                #print(f'Output: {output_dictionary}')
                LOGGER.info("Sending prediction to kafka topic for model: %s", self.model_name)

                # Send out
                output.send_out(timestamp=timestamp,
                                value=output_dictionary,
                                suggested_value = None,
                                algorithm= 'GDB model')


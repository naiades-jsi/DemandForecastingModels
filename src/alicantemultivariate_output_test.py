from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
from datetime import datetime
import pandas as pd

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))


data = pd.read_csv('./data/DataForModels/Multivariate/data_Autobus.csv')
scaled_data = pd.read_csv('./data/DataForModels/Multivariate/scaled_data_Autobus.csv')

test_component = np.array(data[int(0.8*len(data))+1:])
scaled_test_component = np.array(scaled_data[int(0.8*len(scaled_data))+1:])

timesteps = 24
n_future = 48
ftr_vector = np.array([test_component[t:t+timesteps] for t in range(0, len(test_component)-timesteps)]).tolist()
scaled_ftr_vector = np.array([scaled_test_component[t:t+timesteps] for t in range(0, len(scaled_test_component)-timesteps)]).tolist()
data = {"ftr_vector" : ftr_vector[-n_future:],
        "scaled_ftr_vector" : scaled_ftr_vector[-n_future:],
        "timestamp": str(datetime.now())}
print(data)
print(np.shape(ftr_vector[-n_future:]))
print(np.shape(scaled_ftr_vector[-n_future:]))

producer.send('prediction_alicante_multi', value=data)
sleep(1)

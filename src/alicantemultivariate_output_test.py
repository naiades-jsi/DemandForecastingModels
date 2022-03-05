from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
from datetime import datetime
import pandas as pd
import json

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))


data = pd.read_csv('./data/DataForModels/Multivariate/data_Autobus.csv')
scaled_data = pd.read_csv('./data/DataForModels/Multivariate/scaled_data_Autobus.csv')

test_component = np.array(data[int(0.8*len(data))+1:])
scaled_test_component = np.array(scaled_data[int(0.8*len(scaled_data))+1:])

timesteps = 48
n_days = 7
n_features = 6

ftr_vector = np.array([test_component[t:t+timesteps] for t in range(0, len(test_component)-timesteps)]).tolist()
scaled_ftr_vector = np.array([scaled_test_component[t:t+timesteps] for t in range(0, len(scaled_test_component)-timesteps)]).tolist()

for i in range(400):
        arr = np.array(ftr_vector[i]).reshape(1, timesteps*n_features, order = 'C')[0]
        arr1 = np.concatenate([arr[0::6], arr[1::6], arr[2::6], arr[3::6], arr[4::6], arr[5::6]])

        data = {"ftr_vector" : list(arr1),
                "timestamp": str(datetime.now())}
        
        print(data['ftr_vector'])

        producer.send('prediction_alicante_multi', value=data)
        sleep(0.1)

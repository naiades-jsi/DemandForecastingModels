from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
from datetime import datetime
import pandas as pd

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))


def feature_vector_normalization(ftr_vector):
    scaled_ftr_vector = (ftr_vector-np.min(ftr_vector))/(np.max(ftr_vector)-np.min(ftr_vector))
    return scaled_ftr_vector

data = pd.read_csv('./data/DataForModels/Univariate/data_Autobus.csv')
Values = data['Values']
values = Values[int(0.8*len(Values))+1:]
test_component = feature_vector_normalization(values)
timesteps = 24
n_future = 48
ftr_vector = np.array([values[t:t+timesteps] for t in range(0, len(values)-timesteps)]).tolist()
scaled_ftr_vector = np.array([test_component[t:t+timesteps] for t in range(0, len(test_component)-timesteps)]).tolist()
data = {"ftr_vector" : ftr_vector[-n_future:],
        "scaled_ftr_vector" : scaled_ftr_vector[-n_future:],
        "timestamp": str(datetime.now())}
print(data)
print(np.shape(scaled_ftr_vector[-n_future:]))

producer.send('topic', value=data)
sleep(1)
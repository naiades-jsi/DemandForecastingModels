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

data = pd.read_csv('../data/DataForModels/Braila/Flow.csv')
Values = data['Values']
values = Values[int(0.8*len(Values))+1:]
test_component = feature_vector_normalization(values)
timesteps = 36
n_days = 7
n_features = 1


ftr_vector = np.array([values[t:t+timesteps] for t in range(0, len(values)-timesteps)]).tolist()
scaled_ftr_vector = np.array([test_component[t:t+timesteps] for t in range(0, len(test_component)-timesteps)]).tolist()

#send 400 fvs (the algorithm needs to get timesteps*n_days fvs before it starts to produce a result)
for i in range(400):
    # send fvs of shape (1, timesteps*n_features)

    arr = np.array(ftr_vector[i]).reshape(1, timesteps*n_features, order = 'C')[0]

    data = {"ftr_vector" : list(arr),
            "timestamp": str(datetime.now())
            # no scaled_ftr_vector_here!
            }
    
    #print(data['ftr_vector'])
    producer.send('prediction_braila_uni', value=data)
    print(i)
    sleep(0.1)
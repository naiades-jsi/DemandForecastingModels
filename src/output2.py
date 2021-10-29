import csv
import json
from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
from datetime import datetime
import random

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))

tab_data = []
for i in range(0,50):
    n = np.random.rand(1,24)
    n = n.tolist()
    tab_data.append(n)
    print(tab_data)
tab_data_csv = []

for e in range(24):
    print(str(e))
    timestamp = e
    ran = float(np.random.normal(0, 0.1))
    print(tab_data[e][0])
    
    if(e%10 == 0):
        ran += 0.4
    data = {"ftr_vector" : tab_data[e][0],
			"timestamp": str(datetime.now())}

    data_csv = {"test_value" : 3 + ran,
                "second": e,
			    "timestamp": str(datetime.now())}
    tab_data_csv.append(data_csv)

    print(data)

    producer.send('input_topic', value=data)
    sleep(1)
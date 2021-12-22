from abc import ABC, abstractmethod
import csv
import json
import sys

from typing import Any, Dict, List
sys.path.insert(0,'./src')
# Algorithm imports
from kafka import KafkaConsumer, TopicPartition
from json import loads
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pandas as pd
import datetime
from src.model import LSTM_model


class ConsumerAbstract(ABC):
    configuration_location: str

    def __init__(self, configuration_location: str = None) -> None:
        self.configuration_location = configuration_location

    @abstractmethod
    def configure(self, con: Dict[Any, Any],
                  configuration_location: str) -> None:
        self.configuration_location = configuration_location

    @abstractmethod
    def read(self) -> None:
        pass

    # rewrites prediction  configuration
    def rewrite_configuration(self, prediction__conf: Dict[str, Any]
                              ) -> None:
        with open(self.configuration_location) as c:
            conf = json.load(c)
            conf["prediction_conf"] = prediction_conf

        with open(self.configuration_location, "w") as c:
            json.dump(conf, c)




class ConsumerKafka(ConsumerAbstract):
    anomalies: List["predictionAbstract"]
    prediction_names: List[str]
    prediction_configurations: List[Any]
    consumer: KafkaConsumer

    def __init__(self, conf: Dict[Any, Any] = None,
                 configuration_location: str = None) -> None:
        super().__init__(configuration_location=configuration_location)
        if(conf is not None):
            self.configure(con=conf)
        elif(configuration_location is not None):
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        if(con is None):
            print("No configuration was given")
            return 

        if("filtering" in con):
            self.filtering = con['filtering']
        else:
            self.filtering = None

        self.topics = con['topics']
        self.consumer = KafkaConsumer(
                        bootstrap_servers=con['bootstrap_servers'],
                        auto_offset_reset=con['auto_offset_reset'],
                        enable_auto_commit=con['enable_auto_commit'],
                        group_id=con['group_id'],
                        value_deserializer=eval(con['value_deserializer']))
        self.consumer.subscribe(self.topics)

        # Initialize a list of prediction algorithms, each for a
        # seperate topic
        self.model_names = con["model_alg"]
        self.model_configurations = con["model_conf"]
        # check if the lengths of configurations, algorithms and topics are
        # the same
        assert (len(self.model_names) == len(self.topics) and
                len(self.topics) == len(self.model_configurations)),\
                "Number of algorithms, configurations and topics does not match"
        self.models = []
        algorithm_indx = 0
        for model_name in self.model_names:
            Model = eval(model_name)
            Model.configure(self.model_configurations[algorithm_indx],
                              configuration_location=self.configuration_location,
                              algorithm_indx=algorithm_indx)
            self.models.append(Model)
            algorithm_indx += 1
            
    def read(self) -> None:
        for message in self.consumer:
            # Get topic and insert into correct algorithm
            #print(message)
            topic = message.topic
            print('topic: ' + str(topic), flush=True)
            algorithm_indx = self.topics.index(topic)
            
            #check if this topic needs filtering
            if(self.filtering is not None and eval(self.filtering[algorithm_indx]) is not None):
                #extract target time and tolerance
                target_time, tolerance = eval(self.filtering[algorithm_indx])
                message = self.filter_by_time(message, target_time, tolerance)

            if message is not None:
                value = message.value
                self.models[algorithm_indx].feature_vector_creation(value)

    def filter_by_time(self, message, target_time, tolerance):
        #convert to timedelta objects

        # Convert unix timestamp to datetime format (with seconds unit if
        # possible alse miliseconds)

        print('filering; timestamp: ' + str(message.value['timestamp']), flush=True)
        try:
            timestamp = pd.to_datetime(message.value['timestamp'], unit="s")
        except(pd._libs.tslibs.np_datetime.OutOfBoundsDatetime):
            timestamp = pd.to_datetime(message.value['timestamp'], unit="ms")

        # timestamp = pd.to_datetime(message.value['timestamp'], unit='s')
        time = timestamp.time()
        target_time = datetime.time(target_time[0], target_time[1], target_time[2])
        tol = datetime.timedelta(hours = tolerance[0], minutes = tolerance[1], seconds = tolerance[2])
        date = datetime.date(1, 1, 1)
        datetime1 = datetime.datetime.combine(date, time)
        datetime2 = datetime.datetime.combine(date, target_time)

        # Return message only if timestamp is within tolerance
        # print((max(datetime2, datetime1) - min(datetime2, datetime1)))
        # print(tol)
        print('razlika: ' + str((max(datetime2, datetime1) - min(datetime2, datetime1))), flush=True)
        if((max(datetime2, datetime1) - min(datetime2, datetime1)) < tol):
            print('filtriral!', flush=True)
            return(message)
        else:
            print('Nisem :(', flush=True)
            return(None)




class ConsumerFile(ConsumerAbstract):
    prediction: "predictionAbstract"
    file_name: str
    file_path: str

    def __init__(self, conf: Dict[Any, Any] = None,
                 configuration_location: str = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(con=conf)
        elif(configuration_location is not None):
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        self.file_name = con["file_name"]
        self.file_path = self.file_name

        # Expects a list but only requires the first element
        self.prediction = eval(con["prediction__alg"][0])
        prediction_configuration = con["prediction__conf"][0]
        self.prediction.configure(prediction_configuration,
                               configuration_location=self.configuration_location,
                               algorithm_indx=0)

    def read(self) -> None:
        if(self.file_name[-4:] == "json"):
            self.read_JSON()
        elif(self.file_name[-3:] == "csv"):
            self.read_csv()
        else:
            print("Consumer file type not supported.")
            sys.exit(1)

    def read_JSON(self):
        with open(self.file_path) as json_file:
            data = json.load(json_file)
            tab = data["data"]
        for d in tab:
            self.prediction.message_insert(d)

    def read_csv(self):
        with open(self.file_path, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)

            header = next(csv_reader)

            try:
                timestamp_index = header.index("timestamp")
            except ValueError:
                timestamp_index = None
            other_indicies = [i for i, x in enumerate(header) if (x != "timestamp")]

            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                d = {}
                if(timestamp_index is not None):
                    timestamp = row[timestamp_index]
                    try:
                        timestamp = float(timestamp)
                    except ValueError:
                        pass
                    d["timestamp"] = timestamp
                ftr_vector = [float(row[i]) for i in other_indicies]

                d["ftr_vector"] = ftr_vector

                self.prediction.message_insert(d)


class ConsumerFileKafka(ConsumerKafka, ConsumerFile):
    prediction: "predictionAbstract"
    file_name: str
    file_path: str

    def __init__(self, conf: Dict[Any, Any] = None,
                 configuration_location: str = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(con=conf)
        elif(configuration_location is not None):
            # Read config file
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any] = None) -> None:
        # File configuration
        self.file_name = con["file_name"]
        self.file_path = "./data/consumer/" + self.file_name

        # Kafka configuration
        self.topics = con['topics']
        self.consumer = KafkaConsumer(
                        bootstrap_servers=con['bootstrap_servers'],
                        auto_offset_reset=con['auto_offset_reset'],
                        enable_auto_commit=con['enable_auto_commit'],
                        group_id=con['group_id'],
                        value_deserializer=eval(con['value_deserializer']))
        self.consumer.subscribe(self.topics)

        # Expects a list but only requires the first element
        self.prediction = eval(con["prediction__alg"][0])
        prediction_configuration = con["prediction__conf"][0]
        self.prediction.configure(prediction_configuration,
                               configuration_location=self.configuration_location,
                               algorithm_indx=0)

    def read(self) -> None:
        ConsumerFile.read(self)
        
        # expects only one topic
        for message in self.consumer:
            value = message.value
            self.prediction.message_insert(value)
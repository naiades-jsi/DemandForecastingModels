from abc import ABC, abstractmethod
import csv
import json
import sys
from src.model import Model
from typing import Any, Dict, List
sys.path.insert(0,'./src')

from kafka import KafkaConsumer, TopicPartition
from json import loads
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pandas as pd
import datetime

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

    # rewrites model configuration
    def rewrite_configuration(self, models: Dict[str, Any]
                              ) -> None:
        with open(self.configuration_location) as c:
            conf = json.load(c)
            conf["models"] = models

        with open(self.configuration_location, "w") as c:
            json.dump(conf, c)




class ConsumerKafka(ConsumerAbstract):
    models: List["Model"]
    model_name: List[str]
    model_configuration: List[Any]
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
        
        self.topics = con['topics']
        self.consumer = KafkaConsumer(bootstrap_servers=con['bootstrap_server'])
        self.consumer.subscribe(self.topics)
        
        # Initialize forecasting algorithm for the respective topic
        self.model_name = con["model_alg"]
        self.model_configuration = con["model_structure"]
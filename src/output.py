from abc import abstractmethod
from abc import ABC
import json
import csv
from json import dumps
import os
import logging
from typing import Any, Dict
from kafka import KafkaProducer
#from kafka.admin import KafkaAdminClient, NewTopic

class OutputAbstract(ABC):
    send_ok: bool

    def __init__(self) -> None:
        pass

    @abstractmethod
    def configure(self, conf: Dict[Any, Any]) -> None:
        if("send_ok" in conf):
            self.send_ok = conf["send_ok"]
        else:
            self.send_ok = True

    @abstractmethod
    def send_out(self, value: Any, suggested_value: Any, status: str, timestamp: Any,
                 status_code: int = None, algorithm: str = "Unknown"
                 ) -> None:
        pass

class KafkaOutput(OutputAbstract):

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        # print(conf)
        if(conf is not None):
            self.configure(conf=conf)
        print("KafkaOutput initialized", flush=True)

    def configure(self, conf: Dict[Any, Any]) -> None:
        super().configure(conf=conf)
        self.output_topic = conf['output_topic']

        self.producer = KafkaProducer(bootstrap_servers="localhost:9092",
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))
        print("KafkaOutput configured", flush=True)

    def send_out(self, suggested_value: Any,status: str = "",
                 timestamp: Any = None, status_code: int = None,
                value: Any = None,
                 algorithm: str = "Unknown") -> None:
        print(self.output_topic, flush=True)
        print(value)
        print(type(value))
        self.producer.send(self.output_topic, value=value)

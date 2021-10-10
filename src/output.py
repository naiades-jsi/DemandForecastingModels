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

    def configure(self, conf: Dict[Any, Any]) -> None:
        super().configure(conf=conf)
        self.node_id = conf['node_id']

        self.producer = KafkaProducer(bootstrap_server=['localhost:9092'])

    def send_out(self, suggested_value: Any,status: str = "",
                 timestamp: Any = None, status_code: int = None,
                value: Any = None,
                 algorithm: str = "Unknown") -> None:

        if(status_code != 1 or self.send_ok):
            # Construct the object to write
            to_write = {"algorithm": algorithm}
            if (value is not None):
                to_write["value"] = value
            if (status != ""):
                to_write["status"] = status
            if (timestamp is not None):
                to_write["timestamp"] = timestamp
            if (status_code is not None):
                to_write["status_code"] = status_code
            if(suggested_value is not None):
                to_write["suggested_value"] = suggested_value
            
            kafka_topic = "predictions_" + str(self.node_id)

            self.producer.send(kafka_topic, value=to_write)

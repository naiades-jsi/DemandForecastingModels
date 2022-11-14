from abc import abstractmethod
from abc import ABC
import json
import csv
from json import dumps
import os
import logging
from typing import Any, Dict
from kafka import KafkaProducer
import logging
#from kafka.admin import KafkaAdminClient, NewTopic

# logging
LOGGER = logging.getLogger("wf-monitor")
logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO)


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
        LOGGER.info("KafkaOutput initialized")

    def configure(self, conf: Dict[Any, Any]) -> None:
        super().configure(conf=conf)
        self.output_topic = conf['output_topic']

        self.producer = KafkaProducer(bootstrap_servers="localhost:9092",
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))
        LOGGER.info("KafkaOutput configured")

    def send_out(self, suggested_value: Any,status: str = "",
                 timestamp: Any = None, status_code: int = None,
                value: Any = None,
                 algorithm: str = "Unknown") -> None:
        LOGGER.info("Sending to topic: %s", self.output_topic)
        LOGGER.info("Length of the vector: %s", value["ftr_vector"])
        self.producer.send(self.output_topic, value=value)

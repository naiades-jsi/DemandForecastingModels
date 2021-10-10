import argparse
import json
import sys
import requests
import threading
import time
import logging
from multiprocessing import Process
from datetime import datetime
from kafka import KafkaConsumer
from kafka import KafkaProducer

# adding src subdirectory
sys.path.insert(0,'./src')

import model as m


def ping_watchdog(process):
    interval = 30 # ping interval in seconds
    url = "localhost"
    port = 5001
    path = "/pingCheckIn/Data adapter"

    while(process.is_alive()):
        print("{}: Pinging.".format(datetime.now()))
        try:
            r = requests.get("http://{}:{}{}".format(url, port, path))
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            logging.warning(e)
        else:
            logging.info('Successful ping at ' + time.ctime())
        time.sleep(interval)

def main():
    logging.basicConfig(filename="event_log.log", format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    parser = argparse.ArgumentParser(description="consumer")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.json",
        help=u"Config file located in ./config/ directory."
    )

    parser.add_argument(
        "-w",
        "--watchdog",
        dest="watchdog",
        action='store_true',
        help=u"Ping watchdog",
    )

    # Display help if no arguments are defined
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit(1)

    # Parse input arguments
    args = parser.parse_args()

    # Read the configuration file
    with open("configuration/" + args.config) as configuration:
        conf = json.load(configuration)

    # Read the topics
    topics = conf["topics"]

    # Create kafka consumer and subscribe to topics
    consumer = KafkaConsumer(bootstrap_servers=conf['bootstrap_servers'])
    consumer.subscribe(topics)
    print("Subscribed to topics: ", topics, flush=True)

    # Read model configurations
    model_configurations = conf["bootstrap_servers"]

    # Initialize models (build NN-s and train models)
    models = []
    for model_configuration in model_configurations:
        model = m.Model(conf=model_configuration)
        models.append(m)

    # Infinite loop through kafka topic
    for msg in consumer:
        try:
            # Extract data from message
            rec = eval(msg.value)
            ftr_vector = rec["frt_vector"]
            timestamp = rec["timestamp"]
            
            # Extract topic name
            topic = msg.topic

            # Find the index of topic and the correct model in the list
            index = topics.index(topic)
            model = models[index]

            # Create dict to insert
            message_value = {
                "timestamp": timestamp,
                "feature_vector": ftr_vector
            }

            # Insert the message into the model (makes prediction, outputs result back to kafka)
            model.message_insert(message_value=message_value)

        except Exception as e:
            print('Consumer error: ' + str(e), flush=True)



if (__name__ == '__main__'):
    main()

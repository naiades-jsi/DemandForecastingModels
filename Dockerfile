# BUILDING: docker build -t <container_name> .
# RUNNING: docker run <container_name> <python_program_path> <config_file_path>
# e.g. docker run --network="host" -d lstm_alicante
FROM ubuntu:20.04
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev
COPY ./requirements.txt /requirements.txt
WORKDIR /
RUN pip3 install -r requirements.txt
COPY . /

# e3ailab/gdb_alicante_univariate_ircai
CMD ["python3", "main.py", "-c", "alicanteunivariateconfig_gdb.json"]
# CMD ["python3", "main.py", "-c", "alicantemultivariateconfig.json"]
# CMD ["python3", "main.py", "-c", "brailaunivariateconfig.json"]

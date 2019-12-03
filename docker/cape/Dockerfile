########################################
# Derived from python image
########################################
FROM python:3.7
MAINTAINER Qitian Zeng <qzeng3@hawk.iit.edu>
########################################
# Install packages needed for building
########################################
RUN apt-get -y update && \
    apt-get -y --no-install-recommends install python3-tk  &&  \
    rm -rf /var/lib/apt/lists/*
########################################
# Clone Cape repository
########################################
RUN git clone --single-branch --branch sigmod-reproducibility https://github.com/IITDBGroup/cape.git /usr/local/cape/
########################################
# Install cape
########################################
RUN cd /usr/local/cape && \
	python3 setup.py install
########################################
# set environment variables for experiments
########################################
# postgres port for experiments
ENV port 5436
# postgres host IP address
ENV pgip 17.17.0.3
########################################
# Start a sleeper script as entrypoint to keep container running
########################################
ADD ./mesleeps.sh /mesleeps.sh
ENTRYPOINT [ "/mesleeps.sh" ]
##CMD [ "help" ]

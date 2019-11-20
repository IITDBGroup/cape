########################################
# Derived from ubuntu distribution
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
# Cape is entrypoint
########################################
ENTRYPOINT [ "capexplain" ]
CMD [ "help" ]

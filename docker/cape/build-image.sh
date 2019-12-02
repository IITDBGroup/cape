#!/bin/bash
sudo docker build -f ./Dockerfile -t iitdbgroup/2019-sigmod-reproducibility-cape:latest .
sudo docker push iitdbgroup/2019-sigmod-reproducibility-cape:latest

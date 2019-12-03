#!/bin/bash
sudo docker build --no-cache -f ./Dockerfile -t iitdbgroup/2019-sigmod-reproducibility-cape:latest .
sudo docker push iitdbgroup/2019-sigmod-reproducibility-cape:latest

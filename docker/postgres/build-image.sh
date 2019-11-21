#!/bin/bash
rm -rf ./cluster
sudo cp -r /home/perm/antiprov_cluster/ ./cluster
sudo docker build -f ./Dockerfile -t iitdbgroup/2019-sigmod-reproducibility-cape-postgres .

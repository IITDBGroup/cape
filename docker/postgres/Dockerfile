########################################
# Derived from postgres container
########################################
FROM postgres:10-alpine
MAINTAINER Qitian Zeng <qzeng3@hawk.iit.edu>
########################################
# Copy in postgres cluster (shutdown postgres first!)
########################################
ADD ./cluster /var/lib/postgresql/data/

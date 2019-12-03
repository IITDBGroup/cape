#!/bin/bash
if [ -z ${port} ];
then
    port=5437;
fi;

if [ -z ${pgip} ];
then
    pgip='localhost';
fi;

capexplain explain -h ${pgip} -P ${port} -u antiprov -d antiprov -p antiprov --ufile ./input/crime.txt --ofile output_crime.txt

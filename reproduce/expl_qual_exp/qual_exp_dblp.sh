#!/bin/bash
#parameters
if [ -z ${port} ];
then
    port=5437;
fi;

if [ -z ${pgip} ];
then
    pgip='localhost';
fi;

capexplain explain -h ${pgip} -P ${port} -u antiprov -d antiprov -p antiprov --ptable dev.pub_large_no_domain --qtable pub_large_no_domain --ufile ./input/dblp.txt --ofile ../experiments/expl_qual_crime.txt

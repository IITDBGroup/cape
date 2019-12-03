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

echo 'Running experiments for Figure 7'

python3 params_exp.py -h ${pgip} -P ${port} --ufile ./input/user_question.csv --rtfile ./experiments/expl_params_top_10_delta_5.txt

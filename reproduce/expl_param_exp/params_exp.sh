#!/bin/bash

echo 'Running experiments for Figure 7'

python3 params_exp.py --ufile ./input/user_question.csv
python3 plot_params.py

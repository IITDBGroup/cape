#!/bin/bash

echo 'Running experiments for Figure 7'

python3 expl_gt.py --ufile ./input/user_question_expl_gt_7.txt
python3 plot_params.py

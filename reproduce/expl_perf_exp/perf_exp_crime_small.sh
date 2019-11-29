#!/bin/bash

echo 'Running experiments for Figure 6 (b)(c)'

mkdir -p ./time_record

pattern_size_crime=('800k' '400k' '200k' '100k' '50k' '25k')

for exp_id in 1 2 3 4 5 6
do
    for topk in 3
    do
        echo "Find top-${topk} explanations for Chicago crime data with ${pattern_size_crime[$topk]} patterns using ExplGen-Naive"
        capexplain explain -u antiprov -d antiprov -p antiprov -P 5436 --ptable dev.crime_subset --qtable crime_subset \
            --ufile ./input/crime_small.csv --ofile output/crime_small_no_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --rtfile ./time_record/crime_no_pruning_top${topk}_exp_${exp_id}.csv
    done
done

for exp_id in 1 2 3 4 5 6
do
    for topk in 3
    do
        echo "Find top-${topk} explanations for Chicago crime data with ${pattern_size_crime[$topk]} patterns using ExplGen-Opt"
        capexplain explain -u antiprov -d antiprov -p antiprov -P 5436 --ptable dev.crime_subset --qtable crime_subset \
            --ufile ./input/crime_small.csv --ofile output/crime_small_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --pruning --rtfile ./time_record/crime_pruning_top${topk}_exp_${exp_id}.csv
    done
done

python3 expl_crime_numpat.py
python3 expl_crime_numatt.py
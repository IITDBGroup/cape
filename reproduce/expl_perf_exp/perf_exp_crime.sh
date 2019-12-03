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

echo 'Running experiments for Figure 6 (b)(c)'

pattern_size_crime=('800k' '400k' '200k' '100k' '50k' '25k')


mkdir -p ../experiments/expl_time_record
mkdir -p ./output

for exp_id in 1 2 3 4 5 6
do
    for topk in 3
    do
        echo "Find top-${topk} explanations for Chicago crime data with ${pattern_size_crime[$topk]} patterns using ExplGen-Naive"
        capexplain explain -h ${pgip} -P ${port} -u antiprov -d antiprov -p antiprov --ptable dev.crime_subset --qtable crime_subset \
            --ufile ./input/crime.csv --ofile output/crime_no_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --rtfile ../experiments/expl_time_record/crime_no_pruning_top${topk}_exp_${exp_id}.csv
    done
done

for exp_id in 1 2 3 4 5 6
do
    for topk in 3
    do
        echo "Find top-${topk} explanations for Chicago crime data with ${pattern_size_crime[$topk]} patterns using ExplGen-Opt"
        capexplain explain -h ${pgip} -P ${port} -u antiprov -d antiprov -p antiprov --ptable dev.crime_subset --qtable crime_subset \
            --ufile ./input/crime.csv --ofile output/crime_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --pruning --rtfile ../experiments/expl_time_record/crime_pruning_top${topk}_exp_${exp_id}.csv
    done
done

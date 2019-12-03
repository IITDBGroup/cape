#!/bin/bash

echo 'Running experiments for Figure 6 (a)'

if [ -z ${port} ];
then
    port=5437;
fi;

if [ -z ${pgip} ];
then
    pgip='localhost';
fi;

pattern_size_dblp=('800k' '400k' '200k' '100k' '40k' '20k' '10k')

mkdir -p ../experiments/expl_time_record
mkdir -p ./output

for exp_id in 1 2 3 4 5 7
do
    for topk in 3 10
    do
        echo "Find top-${topk} explanations for DBLP data with ${pattern_size_dblp[$topk]} patterns using ExplGen-Naive"
        capexplain explain -h ${pgip} -P ${port} -u antiprov -d antiprov -p antiprov --ptable dev.pub_large_no_domain --qtable pub_large_no_domain \
            --ufile ./input/dblp.csv --ofile output/dblp_no_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --rtfile ../experiments/expl_time_record/dblp_no_pruning_top${topk}_exp_${exp_id}.csv
    done
done

for exp_id in 1 2 3 4 5 7
do
    for topk in 3 10
    do
        echo "Find top-${topk} explanations for DBLP data with ${pattern_size_dblp[$topk]} patterns using ExplGen-Opt"
        capexplain explain -h ${pgip} -P ${port} -u antiprov -d antiprov -p antiprov --ptable dev.pub_large_no_domain --qtable pub_large_no_domain \
            --ufile ./input/dblp.csv --ofile output/dblp_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --pruning --rtfile ../experiments/expl_time_record/dblp_pruning_top${topk}_exp_${exp_id}.csv
    done
done
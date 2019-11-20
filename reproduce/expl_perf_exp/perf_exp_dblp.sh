#!/bin/bash

echo 'Running experiments for Figure 6 (a)'

pattern_size_dblp=('800k' '400k' '200k' '100k' '40k' '20k' '10k')

for exp_id in 1 2 3 4 5 7
do
    for topk in 3 10
    do
        echo "Find top-${topk} explanations for DBLP data with ${pattern_size_dblp[$topk]} patterns using ExplGen-Naive"
        capexplain explain -u antiprov -d antiprov -p antiprov -P 5436 \
            --ptable dev.pub_large_no_domain --qtable pub_large_no_domain --ufile ./input/dblp_${exp_id}.txt --ofile output.txt \
            --exp_id $exp_id --expl_topk=$topk
    done
done

for exp_id in 1 2 3 4 5 7
do
    for topk in 3 10
    do
        echo "Find top-${topk} explanations for DBLP data with ${pattern_size_dblp[$topk]} patterns using ExplGen-Opt"
        capexplain explain -u antiprov -d antiprov -p antiprov -P 5436 \
            --ptable dev.pub_large_no_domain --qtable pub_large_no_domain --ufile ./input/dblp_${exp_id}.txt --ofile output.txt \
            --exp_id $exp_id --expl_topk=$topk --pruning
    done
done


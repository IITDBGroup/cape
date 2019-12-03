#!/bin/bash

#parameters
if [ -z ${port} ];
then
    port=5440;
fi;

if [ -z ${pgip} ];
then
    pgip='localhost';
fi;


echo "parameters: port=${port}";

echo 'Running experiments for Table 3 and Table 4'
capexplain explain -h ${pgip} -u antiprov -d antiprov -p antiprov -P 5436 --ptable dev.pub_large_no_domain --qtable pub_large_no_domain --ufile ./expl_qual_exp/dblp.txt --ofile ./expl_qual_exp/output_dblp.txt

echo 'Running experiments for Table 5'
capexplain explain -h ${pgip} -u antiprov -d antiprov -p antiprov -P 5436 --ufile ./expl_qual_exp/input/crime.txt --ofile ./expl_qual_exp/output_crime.txt


echo 'Running experiments for Figure 6 (a)'

pattern_size_dblp=('800k' '400k' '200k' '100k' '40k' '20k' '10k')

mkdir -p ./experiments/expl_time_record
# for exp_id in 1 2 3 4 5 7
for exp_id in 7
do
    for topk in 3 10
    do
        echo "Find top-${topk} explanations for DBLP data with ${pattern_size_dblp[$topk]} patterns using ExplGen-Naive"
        capexplain explain -h ${pgip} -u antiprov -d antiprov -p antiprov -P ${port} --ptable dev.pub_large_no_domain --qtable pub_large_no_domain \
            --ufile ./expl_perf_exp/input/dblp.csv --ofile ./expl_perf_exp/output/dblp_no_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --rtfile ./experiments/expl_time_record/dblp_no_pruning_top${topk}_exp_${exp_id}.csv
    done
done

# for exp_id in 1 2 3 4 5 7
for exp_id in 7
do
    for topk in 3 10
    do
        echo "Find top-${topk} explanations for DBLP data with ${pattern_size_dblp[$topk]} patterns using ExplGen-Opt"
        capexplain explain -h ${pgip} -u antiprov -d antiprov -p antiprov -P ${port} --ptable dev.pub_large_no_domain --qtable pub_large_no_domain \
            --ufile ./expl_perf_exp/input/dblp.csv --ofile ./expl_perf_exp/output/dblp_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --pruning --rtfile ./experiments/expl_time_record/dblp_pruning_top${topk}_exp_${exp_id}.csv
    done
done

# python3 expl_DBLP_numpat.py

echo 'Running experiments for Figure 6 (b)(c)'

pattern_size_crime=('800k' '400k' '200k' '100k' '50k' '25k')

# for exp_id in 1 2 3 4 5 6
for exp_id in 6
do
    for topk in 3
    do
        echo "Find top-${topk} explanations for Chicago crime data with ${pattern_size_crime[$topk]} patterns using ExplGen-Naive"
        capexplain explain -h ${pgip} -u antiprov -d antiprov -p antiprov -P ${port} --ptable dev.crime_subset --qtable crime_subset \
            --ufile ./expl_perf_exp/input/crime.csv --ofile ./expl_perf_exp/output/crime_no_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --rtfile ./experiments/expl_time_record/crime_no_pruning_top${topk}_exp_${exp_id}.csv
    done
done

# for exp_id in 1 2 3 4 5 6
for exp_id in 6
do
    for topk in 3
    do
        echo "Find top-${topk} explanations for Chicago crime data with ${pattern_size_crime[$topk]} patterns using ExplGen-Opt"
        capexplain explain -h ${pgip} -u antiprov -d antiprov -p antiprov -P ${port} --ptable dev.crime_subset --qtable crime_subset \
            --ufile ./expl_perf_exp/input/crime.csv --ofile ./expl_perf_exp/output/crime_pruning_top${topk}_exp_${exp_id}.txt \
            --exp_id $exp_id --expl_topk=$topk --pruning --rtfile ./experiments/expl_time_record/crime_pruning_top${topk}_exp_${exp_id}.csv
    done
done

# python3 expl_crime_numpat.py
# python3 expl_crime_numatt.py


echo 'Running experiments for Figure 7'
python3 ./expl_param_exp/params_exp.py -h ${pgip} -P ${port} --ufile ./expl_param_exp/input/user_question.txt --rtfile ./experiments/expl_params_top_10_delta_5.txt

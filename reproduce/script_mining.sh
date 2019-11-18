#!/bin/bash

#first create/empty all the csv files
echo "">experiments/crime_fd_on_off.csv
echo "">experiments/crime_num_att.csv
echo "">experiments/crime_size.csv
echo "">experiments/dblp_size.csv
rm experiments/crime_bar.csv

#start running

algorithms='cube share_grp optimized'

echo 'Running experiments for Figure 3 (a)'
for algo in 'naive' $algorithms
do
    for num in {4..11}
    do
        echo "mining crime_exp_${num} with $algo"
        #capexplain mine -u antiprov -d antiprov -p antiprov -P 5436 -t crime_exp_$num --algorithm $algo --numeric year --show-progress False --experiment 'num_attribute' --csv 'experiments/crime_num_att.csv'
    done
done

echo 'Running experiments for Figure 3 (b)'
for algo in $algorithms
do
    for size in '10000' '18000' '32000' '56000' '100000' \
        '180000' '320000' '560000' '1000000'
    do
        echo "mining crime_$size with $algo"
        #capexplain mine -u antiprov -d antiprov -p antiprov -P 5436 -t crime_$size --algorithm $algo --numeric year --show-progress False --experiment 'size' --csv 'experiments/crime_size.csv'
    done
done

echo 'Running experiments for Figure 3 (c)'
for algo in $algorithms
do
    for size in '10000' '18000' '32000' '56000' '100000' \
        '180000' '320000' '560000' '1000000'
    do
        echo "mining pub_$size with $algo"
        #capexplain mine -u antiprov -d antiprov -p antiprov -P 5436 -t pub_$size --algorithm $algo --numeric year,pubcount --summable pubcount --show-progress False --experiment 'size' --csv 'experiments/dblp_size.csv'
    done
done

echo 'Figure 4 uses the same experiment data as Figure 3 (a)'
#cp crime_num_att.csv crime_bar.csv

echo 'Running experiments for Figure 5'
for algo in $algorithms
do
    for size in '10000' '50000' '100000'
    do
        echo "mining crime_fd_$size with $algo for both fd detection on and off"
        #capexplain mine -u antiprov -d antiprov -p antiprov -P 5436 -t crime_fd_$size --algorithm $algo --numeric year --show-progress False --experiment 'fd' --csv 'experiments/crime_fd_on_off.csv'
    done
done

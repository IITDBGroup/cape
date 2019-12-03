#!/bin/bash

#parameters
if [ -z ${port} ];
then
    port=5437;
fi;


if [ -z ${lsup} ];
then
    lsup=15;
fi;

if [ -z ${gsup} ];
then
    gsup=15;
fi;

if [ -z ${rep} ];
then
    rep=1;
fi;

if [ -z ${pgip} ];
then
    pgip='localhost';
fi;


echo "parameters: port=${port}, lsup=${lsup}, gsup=${gsup}, rep=${rep}";

#files
FILE3A='experiments/crime_num_att.csv';
FILE3B='experiments/crime_size.csv';
FILE3C='experiments/dblp_size.csv';
FILE5='experiments/crime_fd_on_off.csv'


#convenient function for mining
cape_mine() {
    if grep -Fq "${algo},${!1}" $3;
    then
        echo "result exists in output, skip experiment";
    else
        echo "mining $2 with $algo";
        capexplain mine -h $pgip -u antiprov -d antiprov -p antiprov -P ${port} -t $2 --algorithm $algo --local-support $lsup --global-support $gsup --show-progress False --experiment $1 --rep $rep --csv $3;
    fi;
    cp -nR experiments/. $OUTPUTDIR;
}


#start running
algorithms='cube share_grp optimized';


#3a
echo 'Running experiments for Figure 3 (a)'
for algo in $algorithms
do
    for num_attribute in {4..11}
    do
        cape_mine 'num_attribute' crime_exp_${num_attribute} ${FILE3A};
    done
done
algo='naive';
for num_attribute in {4..7}
do
    cape_mine 'num_attribute' crime_exp_${num_attribute} ${FILE3A};
done


#3b
echo 'Running experiments for Figure 3 (b)'
for algo in $algorithms
do
    for size in '10000' '18000' '32000' '56000' '100000' \
        '180000' '320000' '560000' '1000000'
    do
        cape_mine 'size' crime_${size} ${FILE3B};
    done
done


#3c
echo 'Running experiments for Figure 3 (c)'
for algo in $algorithms
do
    for size in '10000' '18000' '32000' '56000' '100000' \
        '180000' '320000' '560000' '1000000'
    do
        cape_mine 'size' pub_${size} ${FILE3C};
    done
done


#4
echo 'Figure 4 uses the same experiment data as Figure 3 (a), copying'
cp experiments/crime_num_att.csv experiments/crime_bar.csv


#5
echo 'Running experiments for Figure 5'
algo='optimized'
for size in '10000' '50000' '100000'
do
    cape_mine 'fd' crime_fd_${size} ${FILE5}
done

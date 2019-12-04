#!/bin/bash

OUTPUTDIR=/experiment_results;

mkdir -p $OUTPUTDIR

cp -nR $OUTPUTDIR/. experiments

source mining.sh
source explanation.sh

make -C experiments

cp -R experiments/. $OUTPUTDIR

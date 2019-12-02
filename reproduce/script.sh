#!/bin/bash

OUTPUTDIR=/experiment_results;

mkdir -p $OUTPUTDIR

cp -nR $OUTPUTDIR/ experiments

OUTPUTDIR=$OUTPUTDIR ./mining.sh
#./script_explaining.sh

make -C experiments

cp -nR experiments/ $OUTPUTDIR

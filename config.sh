#!/usr/bin/env bash

epoch=$1
device=0
if [ $# -ge 2 ]; then
    device=$2
fi

exp=''
if [ $# -ge 3 ]; then
    exp=$3
fi

DATA='data' # input dir
OUT='out' # output dir

DATA_BIN=$OUT/data_bin
DATA_RAW=$OUT/data_raw
mkdir -p $DATA_BIN
mkdir -p $DATA_RAW

MODELS=$OUT/models$exp
RESULT=$OUT/result$exp
mkdir -p $MODELS
mkdir -p $RESULT

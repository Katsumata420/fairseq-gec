#!/usr/bin/env bash

# TODO: multiprocess and split
set -e
set -x

DATA=$1
mkdir -p data_art

# generate data with noise
python noise_data.py -e 1 -s 9182 -d $DATA
python noise_data.py -e 2 -s 78834 -d $DATA
python noise_data.py -e 3 -s 5101 -d $DATA
python noise_data.py -e 4 -s 33302 -d $DATA
python noise_data.py -e 5 -s 781 -d $DATA
python noise_data.py -e 6 -s 1092 -d $DATA
python noise_data.py -e 7 -s 10688 -d $DATA
python noise_data.py -e 8 -s 50245 -d $DATA
python noise_data.py -e 9 -s 71187 -d $DATA

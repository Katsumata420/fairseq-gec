#!/usr/bin/env bash
source ./config.sh

set -e
set -x

# set copy params
copy_params='--copy-ext-dict'

# set common params between train/test
common_params='--source-lang src --target-lang trg  
 --padding-factor 1 
 --srcdict ./bpe_noise11110/vocab.txt 
 --joined-dictionary 
'

for e in {1..1}; do
    echo $epoch

    trainpref=$DATA/train_1b_$epoch
    validpref=$DATA/valid_$epoch

    # preprocess train/valid
    python preprocess.py \
    $common_params \
    $copy_params \
    --trainpref $trainpref \
    --validpref $validpref \
    --destdir ${DATA_BIN}_art_$epoch \
    --output-format binary \
    --alignfile $trainpref.forward \
    | tee $OUT/data_art_bin_$epoch.log

done

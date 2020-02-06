#!/usr/bin/env bash
source ./config.sh

set -e
set -x

rm -rf $DATA_BIN
rm -rf $DATA_RAW

# set copy params
copy_params='--copy-ext-dict'

# set common params between train/test
common_params='--source-lang src --target-lang trg 
--padding-factor 1 
--srcdict ./bpe_noise40/vocab.txt 
--joined-dictionary 
'

trainpref=$DATA/lang8_rulec_russian.tok.clean.bpe_$epoch
# trainpref=$DATA/lang8_nucle_en.tok.clean.bpe_$epoch
validpref=$DATA/valid_$epoch
testpref=$DATA/test_$epoch

# preprocess train/valid
python preprocess.py \
$common_params \
$copy_params \
--trainpref $trainpref \
--validpref $validpref \
--destdir ${DATA_BIN}_ori_$epoch \
--output-format binary \
--alignfile $trainpref.forward \
| tee $OUT/data_bin.log

# preprocess test
python preprocess.py \
$common_params \
$copy_params \
--testpref $testpref \
--destdir ${DATA_RAW}_ori_$epoch \
--output-format raw \
| tee $OUT/data_raw.log

mv ${DATA_RAW}_ori_$epoch/test.src-trg.src ${DATA_RAW}_ori_$epoch/test.src-trg.src.old

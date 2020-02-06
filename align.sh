source ./config.sh
mkdir data_align_no_noise

# trainpref='bpe_noise11110/lang8_rulec_russian.tok.clean.bpe.clean'
trainpref='bpe_noise11110/lang8_nucle_en.tok.bpe.clean'
# trainpref=bpe_no_noise/no_boise_train.bpe.clean
# trainpref='data/valid'
echo "$trainpref.src, trg"

python scripts/build_sym_alignment.py \
 --fast_align_dir /work/katsumata/UMT/monoses/third-party/fast_align/build/ \
 --mosesdecoder_dir /work/katsumata/UMT/mosesdecoder \
 --source_file $trainpref.src --target_file $trainpref.trg \
 --output_dir data_align_no_noise 

cp data_align_no_noise/align.forward $trainpref.forward &
cp data_align_no_noise/align.backward $trainpref.backward &
wait

rm -rf data_align_no_noise

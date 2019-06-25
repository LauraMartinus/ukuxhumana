#!/bin/bash 

###################################################################
#Script Name	: run_t2t_train.sh                                                                                              
#Description	: Runs transformer from tensor2tensor on english-tswana dataset                                                                                
#Args           :                                                                                           
#Author       	:Jade Abbott                                              
#Email         	:jade.zoe.abbott@gmail.com                                           
###################################################################

#pip install -r requirements.txt

# usage: ./run_t2t_train.sh translate_enaf_rma ./clean/en_af enaf_parallel af

problem_name=$1
original_data_dir=$2
original_data_prefix=$3
source=en
target=$4
STORAGE_BUCKET=gs://ukuxhumana-1
DATA_DIR=$STORAGE_BUCKET/translation
TMP_DIR=/home/jade/tmp
echo "============ T2T Training =========="
echo " - Problem Name: $problem_name"
echo " - DATA_DIR: $original_data_dir"
echo " - DATA_PREFIX: $original_data_prefix"
echo " - Source Language: $source"
echo " - Target Language: $target"
echo "   "

mkdir -p "$TMP_DIR/t2t/$problem_name/data"
mkdir -p "$TMP_DIR/t2t/$problem_name/output"
mkdir -p "$TMP_DIR/t2t/$problem_name/tmp"
mkdir -p "$TMP_DIR/t2t/$problem_name/results"

# This just speeds up things since no need to download
cp $original_data_dir/$original_data_prefix.train.* $TMP_DIR/t2t/$problem_name/tmp/
cp $original_data_dir/$original_data_prefix.dev.* $TMP_DIR/t2t/$problem_name/tmp/
cp $original_data_dir/$original_data_prefix.test.*   $TMP_DIR/t2t/$problem_name/tmp/
TPU_NAME=jade-zoe-abbott

# Do training
t2t-trainer \
  --generate_data \
  --data_dir=$DATA_DIR/t2t/$problem_name/data \
  --output_dir=$DATA_DIR/t2t/$problem_name/output \
  --tmp_dir=$TMP_DIR/t2t/$problem_name/tmp \
  --problem=$problem_name \
  --model=transformer \
  --hparams_set=transformer_tpu \
  --train_steps=125000 \
  --eval_steps=100 \
  --t2t_usr_dir=~/ukuxhumana/t2t/problems/ \
  --use_tpu=true \
  --cloud_tpu_name=$TPU_NAME

decode_from=$DATA_DIR/t2t/$problem_name/data/$original_data_prefix.test.$source
decode_to=$DATA_DIR/t2t/$problem_name/results/$original_data_prefix.test.$target
reference=$DATA_DIR/t2t/$problem_name/data/$original_data_prefix.test.$target

#gsutil cp $TMP_DIR/t2t/$problem_name/tmp/$original_data_prefix.test.* $DATA_DIR/t2t/$problem_name/data/
#echo "============ T2T Decoding =========="
# Do decoding
#t2t-decoder \
#  --data_dir=$DATA_DIR/t2t/$problem_name/data \
#  --problem=$problem_name \
#  --model=transformer \
#  --hparams_set=transformer_tpu \
#  --output_dir=$DATA_DIR/t2t/$problem_name/output \
#  --decode_hparams="beam_size=4,alpha=0.6" \
#  --decode_from_file=$decode_from \
#  --decode_to_file=$decode_to \
#  --use_tpu=true \
#  --cloud_tpu_name=jade-zoe-abbott \
#  --t2t_usr_dir=/home/jade/ukuxhumana/t2t/problems/ 

#gsutil cp $reference $TMP_DIR/t2t/$problem_name/data
#gsutil cp $decode_to $TMP_DIR/t2t/$problem_name/results
#echo "============ T2T BLEU =========="
#t2t-bleu \
#    --translation=$TMP_DIR/t2t/$problem_name/results/$original_data_prefix.test.$target
#    --reference=$TMP_DIR/t2t/$problem_name/data/$original_data_prefix.test.$target

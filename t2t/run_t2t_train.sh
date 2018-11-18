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

echo "============ T2T Training =========="
echo " - Problem Name: $problem_name"
echo " - DATA_DIR: $original_data_dir"
echo " - DATA_PREFIX: $original_data_prefix"
echo " - Source Language: $source"
echo " - Target Language: $target"
echo "   "

mkdir -p "/tmp/t2t/$problem_name/data"
mkdir -p "/tmp/t2t/$problem_name/output"
mkdir -p "/tmp/t2t/$problem_name/tmp"
mkdir -p "/tmp/t2t/$problem_name/results"

# This just speeds up things since no need to download
cp $original_data_dir/$original_data_prefix.train.* /tmp/t2t/$problem_name/tmp
cp $original_data_dir/$original_data_prefix.dev.* /tmp/t2t/$problem_name/tmp
cp $original_data_dir/vocab*   /tmp/t2t/$problem_name/tmp

# Do training
t2t-trainer \
  --generate_data \
  --data_dir=/tmp/$problem_name/data \
  --output_dir=/tmp/$problem_name/output \
  --tmp_dir=/tmp/$problem_name/tmp \
  --problem=$problem_name \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --train_steps=125000 \
  --eval_steps=100 \
  --t2t_usr_dir=./t2t/problems/ \
  --warm_start_from=45000 


decode_from=$original_data_dir/$original_data_prefix.test.$source
decode_to=/tmp/t2t/$problem_name/results/$original_data_prefix.test.$target
reference=$original_data_dir/$original_data_prefix.test.$target

# Do decoding
t2t-decoder \
  --data_dir=/tmp/t2t/$problem_name/data \
  --problem=$problem_name \
  --model='transformer' \
  --hparams_set='transformer_base_single_gpu' \
  --output_dir=/tmp/t2t/$problem_name/output \
  --decode_hparams="beam_size=4,alpha=0.6" \
  --decode_from_file=$decode_from \
  --decode_to_file=$decode_to


# BLEU scoring
t2t-bleu \
    --translation=$decode_to
    --reference=$reference
#!/bin/bash 

###################################################################
#Script Name	: run_t2t_train.sh                                                                                              
#Description	: Runs transformer from tensor2tensor on english-tswana dataset                                                                                
#Args           :                                                                                           
#Author       	:Jade Abbott                                              
#Email         	:jade.zoe.abbott@gmail.com                                           
###################################################################

pip install -r requirements.txt

problem_dir=$1
problem_file=$2
data_file=$3
mkdir "/tmp/t2t"
mkdir "/tmp/t2t/$1"
mkdir "/tmp/t2t/$1/data"
mkdir "/tmp/t2t/$1/output"
mkdir "/tmp/t2t/$1/tmp"

# This just speeds up things since no need to download
cp clean/$1/$3.train.* /tmp/t2t/$1/data
cp clean/$1/$3.dev.* /tmp/t2t/$1/data

python $2 -u

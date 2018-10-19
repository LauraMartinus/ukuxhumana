#!/bin/bash 

###################################################################
#Script Name	: run_t2t_train.sh                                                                                              
#Description	: Runs transformer from tensor2tensor on english-tswana dataset                                                                                
#Args           :                                                                                           
#Author       	:Jade Abbott                                              
#Email         	:jade.zoe.abbott@gmail.com                                           
###################################################################

pip install -r requirements.txt

mkdir "/tmp/t2t"
mkdir "/tmp/t2t/data"
mkdir "/tmp/t2t/output"
mkdir "/tmp/t2t/tmp"

# This just speeds up things since no need to download
cp data/eng_tswane/entn_parallel.train.* /tmp/t2t/data
cp data/eng_tswane/entn_parallel.dev.* /tmp/t2t/data

python t2t_entn_problem.py

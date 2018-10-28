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
mkdir "/tmp/t2t/enzu"
mkdir "/tmp/t2t/enzu/data"
mkdir "/tmp/t2t/enzu/output"
mkdir "/tmp/t2t/enzu/tmp"

# This just speeds up things since no need to download
cp data/en_zu/enzn_parallel.train.* /tmp/t2t/enzu/data
cp data/en_zu/enzn_parallel.dev.notest.* /tmp/t2t/enzu/data

python t2t_enzu_problem.py -u

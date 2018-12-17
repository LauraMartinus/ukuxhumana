# For reverse to be true then set this
rev='_rev'
#rev=''

./run_t2t_train$rev.sh translate_enaf_rma ./clean/en_af enaf_parallel af | tee translate_enaf_rma$rev.results
./run_t2t_train$rev.sh translate_ennso_rma ./clean/en_nso ennso_parallel nso | tee translate_ennso_rma$rev.results
./run_t2t_train$rev.sh translate_entn_rma ./clean/en_tn entn_parallel tn | tee translate_entn_rma$rev.results
./run_t2t_train$rev.sh translate_ents_rma ./clean/en_ts ents_parallel ts | tee translate_ents_rma$rev.results
./run_t2t_train$rev.sh translate_enzu_rma8k ./clean/en_zu enzu_parallel zu | tee translate_enzu_rma$rev.results
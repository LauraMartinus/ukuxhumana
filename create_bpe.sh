
#!/bin/bash

#languages=(zu af ts tn nso)
languages=(ts tn nso)
root_folder=clean
bpe_folder=bpe
vocab_size=$1
echo ${languages[@]}
echo "vocab_size $vocab_size"

for t in ${languages[@]}; do
    lang_folder="$PWD/$root_folder/en_$t"
    dest_folder="$PWD/$bpe_folder/en_$t"
    echo ${lang_folder}
    echo ${dest_folder}

    #train files
    L1_train="$lang_folder/en${t}_parallel.train.en"
    L2_train="$lang_folder/en${t}_parallel.train.${t}"
    L1_output_train="$dest_folder/en${t}_parallel.$vocab_size.train.en"
    L2_output_train="$dest_folder/en${t}_parallel.$vocab_size.train.${t}"

    #dev files
    L1_dev="$lang_folder/en${t}_parallel.dev.en"
    L2_dev="$lang_folder/en${t}_parallel.dev.${t}"
    L1_output_dev="$dest_folder/en${t}_parallel.$vocab_size.dev.en"
    L2_output_dev="$dest_folder/en${t}_parallel.$vocab_size.dev.${t}"


    #test files
    L1_test="$lang_folder/en${t}_parallel.test.en"
    L2_test="$lang_folder/en${t}_parallel.test.${t}"
    L1_output_test="$dest_folder/en${t}_parallel.$vocab_size.test.en"
    L2_output_test="$dest_folder/en${t}_parallel.$vocab_size.test.${t}"

    model_prefix="$PWD/bpe/en_${t}/bpe.$vocab_size"
    codes_file="$model_prefix.codes"
    vocab_file="$model_prefix.vocab"

    # train
    echo "Training subword nmt for en_$t"
    cat $L1_train $L2_train | subword-nmt learn-bpe --symbols $vocab_size -o $codes_file
    subword-nmt apply-bpe -c $codes_file < $L1_train > $L1_output_train
    subword-nmt apply-bpe -c $codes_file < $L2_train > $L2_output_train   

    subword-nmt apply-bpe -c $codes_file < $L1_dev > $L1_output_dev
    subword-nmt apply-bpe -c $codes_file < $L2_dev > $L2_output_dev   

    subword-nmt apply-bpe -c $codes_file < $L1_test > $L1_output_test
    subword-nmt apply-bpe -c $codes_file < $L2_test > $L2_output_test   


    cat $L1_output_train $L2_output_train | subword-nmt get-vocab  | cut -f1 -d ' '  > $vocab_file
    # convert

    tar -C $dest_folder -zcvf "$dest_folder/en_${t}.train.tar.gz" "en${t}_parallel.$vocab_size.train.en" "en${t}_parallel.$vocab_size.train.${t}"
    tar -C $dest_folder -zcvf "$dest_folder/en_${t}.dev.tar.gz" "en${t}_parallel.$vocab_size.dev.en" "en${t}_parallel.$vocab_size.dev.${t}"
    


    echo "Done with en_$t"
done

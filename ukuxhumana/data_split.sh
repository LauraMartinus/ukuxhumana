#!/usr/bin/env bash

IFS='.' read -ra name <<< "$1"

pref=${name[0]}
suf=${name[1]}
split -l $[ $(wc -l $1|cut -d" " -f1)*70/100] $1
echo "Done..."
if [ -f xaa ]; then
    train_name="$pref.train.$suf"
    dev_name="$pref.dev.$suf" 
    test_name="$pref.test.$suf"
    echo $train_name $dev_name $test_name
    DIR=$(dirname $name)
    echo $DIR
    mv xaa $train_name

    head -3000 xab > $test_name
    tail -n +3001 xab > $dev_name

else 
    echo "Files were not created."
fi

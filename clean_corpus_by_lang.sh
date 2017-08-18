#!/usr/bin/env bash

input_dir="$1"
output_dir="$2"

lang=${input_dir##*/}
mkdir -p $output_dir
for file in $input_dir/*.txt 
do
        name=${file##*/}
        ./tools/tools/split-sentences.perl -l $lang < $file | sed -e '/^<.*>/d' > $output_dir/$name
done

#!/usr/bin/env bash

input_dir="$1"
output_dir="$2"

for folder in $(find $input_dir -mindepth 1 -maxdepth 1 -type d)
do
    lang=${folder##*/}
    mkdir -p $output_dir/$lang
    for file in $folder/*.txt 
    do
        name=${file##*/}
        sed '/^<.*>$/d' $file > $output_dir/$lang/$name
    done
done

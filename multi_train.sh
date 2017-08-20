#!/usr/bin/env bash

beg=$1
step=$2
end=$3

echo "Training sizes (per language):  "
echo "$(seq $beg $step $end)"
for n in $(seq $beg $step $end)
do
    python3 train.py ./data/de-xmled-europarl ./classifiers/$n.classifier $n
done

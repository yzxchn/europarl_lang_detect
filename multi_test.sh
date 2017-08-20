#!/usr/bin/env bash

beg=$1
step=$2
end=$3

echo "Testing sizes (per language):  "
echo "$(seq $beg $step $end)"
for n in $(seq $beg $step $end)
do
    echo "Testing classifier trained with $n documents per language."
    python3 evaluate.py ./data/europarl.test ./classifiers/$n.classifier > ./results/$n.text
done

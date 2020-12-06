#!/usr/bin/env bash

NUM=20
#for i in {1..$NUM}; do
for i in {1..20}; do
  SEED=$RANDOM
  python text_DCN.py --data_dir data/yahoo_answers/ --n_clusters 10 --seed $SEED --verbose
  echo $SEED
done

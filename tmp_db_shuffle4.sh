#!/usr/bin/env bash

NUM=20
#for i in {1..$NUM}; do
for i in {1..20}; do
  SEED=$RANDOM
  python text_DCN_shuffle.py --data_dir data/dbpedia/ --n_clusters 14 --seed $SEED --init_num 100 --verbose
  echo $SEED
done

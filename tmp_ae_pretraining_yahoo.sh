#!/usr/bin/env bash

corpora_id=2
for i in {1..2}; do
  python ae_pretraining.py --corpora_id $corpora_id --feat_id $i --lr 0.1 --id 0
done

#!/usr/bin/env bash

for i in {0..2}; do
  python ub_kmeans_infersent.py --corpora_id $i;
done

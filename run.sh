#!/usr/bin/env bash

for i in {0..2}; do
  python ub_gsdmm.py --corpora_id $i
done


# for i in {0..2}; do
  # python ub_tfidf.py --corpora_id $i
  # python ub_topics.py --corpora_id $i
# done

#!/usr/bin/env bash

for i in {0..2}; do
  python extract_elmo_text_features.py --corpora_id $i
done

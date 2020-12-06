#!/usr/bin/env bash

python extract_text_features.py --data_dir 'data/dbpedia/'
python extract_text_features.py --data_dir 'data/yahoo_answers/'
# python extract_text_features.py --data_dir 'data/TREC/' --model_id 0 --layer_norm
python extract_text_features.py --data_dir 'data/ag_news/'
# python extract_text_features.py --data_dir 'data/20_ng/' --model_id 0 --layer_norm --split_sents

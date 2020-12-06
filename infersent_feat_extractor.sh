#!/usr/bin/env bash

python extract_infersent_text_features.py --data_dir 'data/dbpedia/'
python extract_infersent_text_features.py --data_dir 'data/yahoo_answers/'
python extract_infersent_text_features.py --data_dir 'data/ag_news/'

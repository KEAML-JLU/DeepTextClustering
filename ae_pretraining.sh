#!/usr/bin/env bash

python ae_pretraining.py --corpora_id 0 --feat_id 3 --lr 0.1 --id 323
python ae_pretraining.py --corpora_id 1 --feat_id 3 --lr 10 --id 323
python ae_pretraining.py --corpora_id 2 --feat_id 3 --lr 1 --id 323

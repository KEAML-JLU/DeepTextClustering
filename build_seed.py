import random
import csv
import argparse
import os
from collections import defaultdict
from utils import load_csv_corpus, initialize_environment
from config import cfg


def get_args():
    parser = argparse.ArgumentParser(description='Random Select Seed for semi-cluster')
    parser.add_argument('--data_dir', type=str, default='data/dbpedia/', help='directory of dataset')
    parser.add_argument('--seed', type=int, default=cfg.RNG_SEED, help='random seed')
    parser.add_argument('--seed_num', type=int, default=100, help='the number of seed for each class')
    parser.add_argument('--verbose', help='whether to print log', action='store_true')
    args = parser.parse_args()
    return args

args = get_args()
data_dir = args.data_dir
random_seed = args.seed
seed_num = args.seed_num
verbose = args.verbose

initialize_environment(random_seed=random_seed)
_, labels, ids = load_csv_corpus(os.path.join(data_dir, cfg.TRAIN_DATA_NAME+'.csv'))

dic = defaultdict(list)

for tmp_id, tmp_label in zip(ids, labels):
    dic[tmp_label].append(tmp_id)

results = []
for l, tmp_ids in dic.items():
    random.shuffle(tmp_ids)
    tmp_ids = tmp_ids[:seed_num]
    results.extend([(tmp_id, l) for tmp_id in tmp_ids])
results.sort()

seed_path = os.path.join(os.path.join(data_dir, cfg.SEED_FILE_NAME))
with open(seed_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(results)

import argparse
import os
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import csv
import json
import lda

from config import cfg
from utils import load_infersent, load_csv_corpus, infersent_encode_sents, dump_feat

def load_json_data(path):
    sents = []
    labels = []
    with open(path) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            d = json.loads(l)
            sents.append(d['text'])
            labels.append(d['cluster'])
    return sents, np.array(labels)

def load_csv_corpus(path):
    labels = []
    sents = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for item in reader:
            labels.append(int(item[0]))
            tmp_doc = item[1].strip()
            sents.append(tmp_doc)
        ids = range(len(sents))
    return sents, labels, ids

def get_stop_words():
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    return stop_words

def get_tf_idf_feat(sents, max_features=2000):
    stop_words = get_stop_words()
    tfidf = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    feat = tfidf.fit_transform(sents)
    feat = feat.toarray()
    return feat

def get_count_feat(sents, max_features=2000):
    stop_words = get_stop_words()
    ct = CountVectorizer(max_features=max_features, stop_words=stop_words, binary=False)
    feat = ct.fit_transform(sents)
    feat = feat.toarray()
    return feat

def parse_args():
    parser = argparse.ArgumentParser(description='Building Text Representation')
    parser.add_argument('--data_dir', dest='db_dir', type=str, default='data/ag_news', help='directory of dataset')
    parser.add_argument('--model_id', type=int, default=0,
                        help='feature extractor model\'s id (0:Infersent, 1:Tfidf2000, 2:Tfidf5000)')
    args = parser.parse_args()
    return args


def get_feat(infersent, data_path, verbose=True, layer_norm=False, split_sents=True):
    if verbose:
        print('Loading Text Data from {}'.format(data_path))
    train_data, train_labels, ids = load_csv_corpus(data_path)
    if verbose:
        print('Building Vocabulary Table for Infersent by {}'.format(data_path))
    infersent.build_vocab(train_data, tokenize=False)
    if verbose:
        print('Extracting Feat using Infersent')
    train_feat = infersent_encode_sents(infersent, train_data, split_sents=split_sents, layer_norm=layer_norm, verbose=False)
    return train_feat, np.array(train_labels), ids


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.db_dir
    model_id = args.model_id
    assert 0 <= model_id <= 2
    split_sents = True
    layer_norm = False

    train_data_path = os.path.join(data_dir, cfg.TRAIN_DATA_NAME+'.csv')
    # test_data_path = os.path.join(data_dir, cfg.TEST_DATA_NAME+'.csv')
    train_feat_path = os.path.join(data_dir, 'tfidf.h5')

    train_sents, train_labels, _ = load_csv_corpus(train_data_path)
    train_feat = get_tf_idf_feat(train_sents)



    print('Dumping Train Text Feat and Labels into {}'.format(train_feat_path))
    dump_feat(train_feat_path, train_feat, labels=train_labels)

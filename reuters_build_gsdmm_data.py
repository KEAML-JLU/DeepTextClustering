import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import json
import csv
import os
import glob

def get_stop_words():
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    return stop_words

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


def get_gsdmm_copus(path, max_features=2000):
    sents, labels, _ = load_csv_corpus(path)
    sents = [s.lower() for s in sents]
    stop_words = get_stop_words()
    ct = CountVectorizer(max_features=max_features, stop_words=stop_words, binary=False)
    ct.fit(sents)
    feature_set = set(ct.get_feature_names())
    sents = [[w for w in s.split() if w in feature_set] for s in sents]
    corpus = []
    for s, l in zip(sents,labels):
        if len(s) == 0:
            continue
        s = ' '.join(s)
        corpus.append(json.dumps({'text': s,'cluster':l}))
    return corpus

for d_name in glob.glob('data/reuters_*/'):
    print(d_name)
    corpus = get_gsdmm_copus(os.path.join(d_name, 'train.csv'))
    with open(os.path.join(d_name, 'data.gsdmm'), 'w') as f:
        for l in corpus:
            f.write(l + '\n')

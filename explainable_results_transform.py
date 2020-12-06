import pandas as pd
import numpy as np
import csv
from collections import defaultdict

def load_corpus(filename):
    corpus = defaultdict(list)
    with open(filename) as f:
        reader = csv.reader(f)
        for item in reader:
            label = int(item[0])
            sentence = ' '.join(item[1:])
            corpus[label].append(sentence)
    corpus = dict(corpus)
    return corpus

corpus = load_corpus('data/ag_news/train.csv')

def get_infomation(word, cluster_idx, corpus, model_info=''):
    word = word.lower()
    sentences = corpus[cluster_idx]
    sentences = [x.lower().split() for x in sentences]
    sentences = [x for x in sentences if word in x]
    positions = [x.index(word) / len(x) for x in sentences]
    d = pd.Series(positions).describe().to_dict()
    d['word'] = word
    d['cluster_idx'] = cluster_idx
    d['model_info'] = model_info
    return d


class_text = '''World Sports Business Sci/Tech'''

tcre_lm_words = {}
tcre_lm_words[0] = "space inc president company iraq microsoft cup corp scientists prices".split()
tcre_lm_words[1] = "cup league coach team football olympic athens manager formula victory".split()
tcre_lm_words[2] = "prices shares bankruptcy billion quarter accounting jobs fund percent insurance".split()
tcre_lm_words[3] = "microsoft wireless software internet pc google chips windows ibm product".split()

tcre_tfidf_words = {}
tcre_tfidf_words[0] = "quot quote profile chart research saying 700 two yesterday ap".split()
tcre_tfidf_words[1] = "quot quote company microsoft chart profile companies games business billion".split()
tcre_tfidf_words[2] = "quot president iraq officials government min- ister company killed people british".split()
tcre_tfidf_words[3] = "quot games said game company league team season night president".split()

freq_lm_words = {}
freq_lm_words[0] = "said ’s president reuters new ap iraq two us minister".split()
freq_lm_words[1] = "new microsoft software ’s internet company quot said computer service".split()
freq_lm_words[2] = "’s new team night first -- ap one game last".split()
freq_lm_words[3] = "said reuters new company inc. ’s york oil percent prices".split()

results = []
for m_name , words in {'tcre_lm':tcre_lm_words, 'tcre_tfidf':tcre_tfidf_words, 'freq_lm':freq_lm_words}.items():
    for k in words.keys():
        for w in words[k]:
            info = get_infomation(w, k, corpus, model_info=m_name)
            results.append(info)
all_info = pd.DataFrame(results)
all_info = all_info.dropna()
all_info.to_csv('1.csv', index=False)


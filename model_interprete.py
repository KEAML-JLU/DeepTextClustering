import nltk
import string
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from pymongo import MongoClient

class MongoProxy(object):
    def __init__(self):
        self.client = MongoClient('192.168.1.107',27017)
        self.db = self.client.cluster_db

    def close(self):
        self.client.close()

    def get_pred(self, corpora, feat_name, collection_name='elmo_results'):
        collection = self.db[collection_name]
        r = collection.find_one({'corpora':corpora, 'feat_name':feat_name})
        if 'best_pred' in r:
            pred = r['best_pred']
            return pred
        return None

    def insert_result(self, corpora, feat_name, result, label='int'):
        collection = self.db['interpret']
        r = collection.insert_one({'corpora':corpora, 'feat_name':feat_name, 'result':result, 'label':label})
        

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

def get_stopwords():
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    return set(stopwords)

def get_glove_vocab(filename):
    vocab = list()
    with open(filename) as f:
        for line in f:
            word, _  = line.split(' ', 1) 
            vocab.append(word.strip())
    return set(vocab)

def preprocess_sent(sent, stopwords, vocab=None):
    tokens = sent.lower().split()
    if vocab is not None:
        tokens = [t for t in tokens if t in vocab]
    tokens = [t for t in tokens if t not in stopwords]
    return ' '.join(tokens)

def train_linear_classification(feat, labels):
    lg = LogisticRegression()
    lg.fit(feat, labels)
    return lg

def get_importance_features(lg_model, features_lst, feat_num=20, process_func=lambda x:x):
    coef = lg_model.coef_
    all_idxs = np.argsort(-np.abs(coef), axis=1)
    class_num = all_idxs.shape[0]
    results = []
    for i in range(class_num):
        idx = all_idxs[i]
        feat_names = np.array(features_lst)[idx[:feat_num]].tolist()
        feat_names = process_func(feat_names)
        results.append(feat_names)
    return results

def get_importance_features_count(sents, pred, feat_num=20, label_num=4):
    sents = [s.split() for s in sents]
    results = []
    from collections import Counter
    for i in range(label_num):
        tmp_pred = (pred == i)
        tmp_sents = np.array(sents)[tmp_pred].tolist()
        r = []
        for s in tmp_sents:
            r.extend(s)
        c = Counter(r)
        c = c.items()
        c.sort(key=lambda x:-x[1])
        tmp_f = [x[0] for x in c][:feat_num]
        results.append(tmp_f)
    return results

def get_glove_vec(filename, vocab):
    vecs = []
    words = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            token, vec_str = line.split(' ', 1)
            vec_str = vec_str.strip()
            token = token.strip()
            if token in vocab:
                words.append(token)
                vecs.append(np.array([float(s) for s in vec_str.split()]))
    word_dict = {w: i for i, w in enumerate(words)}
    return np.stack(vecs), word_dict

def split_sents_id(labels):
    results = defaultdict(list)
    for i, l in enumerate(labels):
        results[l].append(i)
    tmp = []
    for i in sorted(results.keys()):
        tmp.append(results[i])
    assert len(tmp) == max(results.keys()) + 1
    return tmp

def semantic_compactness1(feat_list, word_vecs, word_dict):
    def normalize(vec):
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec
    feat_ids = [word_dict[w] for w in feat_list if w in word_dict]
    assert len(feat_ids) == len(feat_list)
    feat_ids = np.array(feat_ids)
    feat_vecs = word_vecs[feat_ids]
    center = np.mean(feat_vecs, axis=0)
    norm_f = normalize(feat_vecs)
    norm_c = normalize(center)
    r = np.dot(norm_f, norm_c)
    return np.mean(r)

def semantic_compactness2(feat_list, word_vecs, word_dict):
    def normalize(vec):
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec
    feat_ids = [word_dict[w] for w in feat_list if w in word_dict]
    assert len(feat_ids) == len(feat_list)
    feat_ids = np.array(feat_ids)
    feat_vecs = word_vecs[feat_ids]
    norm_f = normalize(feat_vecs)
    center = np.mean(norm_f, axis=0)
    norm_c = normalize(center)
    r = np.dot(norm_f, norm_c)
    return np.mean(r)

def semantic_compactness3(feat_list, word_vecs, word_dict):
    def normalize(vec):
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec
    feat_ids = [word_dict[w] for w in feat_list if w in word_dict]
    assert len(feat_ids) == len(feat_list)
    feat_ids = np.array(feat_ids)
    feat_vecs = word_vecs[feat_ids]
    norm_f = normalize(feat_vecs)
    r = max(np.sum(np.dot(norm_f, norm_f.T), axis=1) - 1)
    r = r / (len(feat_list) - 1)
    return r

def semantic_compactness4(feat_list, word_vecs, word_dict):
    def normalize(vec):
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec
    feat_ids = [word_dict[w] for w in feat_list if w in word_dict]
    assert len(feat_ids) == len(feat_list)
    feat_ids = np.array(feat_ids)
    feat_vecs = word_vecs[feat_ids]
    norm_f = normalize(feat_vecs)
    r = np.sum(np.max(np.dot(norm_f, norm_f.T) - 2 * np.eye(len(feat_ids)), axis=1))
    return r

def compute_score(feat_list, word_vecs, word_dict, semantic_score_func):
    pass

if __name__ == '__main__':

    from sklearn.feature_extraction.text import CountVectorizer
    if False:
        glove_path = 'data/paragram_300_sl999.txt'
        glove_path = 'data/glove.840B.300d.txt'
        sents, labels, _ = load_csv_corpus('data/ag_news/train.csv')
        labels = np.array(labels)
        vocab = get_glove_vocab(glove_path)
        stopwords = get_stopwords()
        mongo = MongoProxy()
        pred = mongo.get_pred('ag_news', 'elmo_mean_ln')
        pred = np.array(pred)
        mongo.close()
        sents = [preprocess_sent(s, stopwords, vocab=vocab) for s in sents]
        cv = CountVectorizer(max_features=2000, binary=True, stop_words=stopwords)
        feat = cv.fit_transform(sents)
        feat = feat.toarray()
        lg = train_linear_classification(feat, pred)
        feats_list = cv.get_feature_names()
        selected_feat = get_importance_features(lg, feats_list,feat_num=5)
        word_vecs, word_dict = get_glove_vec(glove_path, set(feats_list))
        split_ids = split_sents_id(pred)
        # score = 0
        score = []
        for i in range(len(split_ids)):
            tmp_score = semantic_compactness1(selected_feat[i], word_vecs, word_dict)
            # tmp_score *= len(split_ids[i])
            # score += tmp_score
            score.append(tmp_score)
        # score /= len(pred)
        print(score)
    else:
        sents, labels, _ = load_csv_corpus('data/ag_news/train.csv')
        labels = np.array(labels)
        stopwords = get_stopwords()
        mongo = MongoProxy()
        pred = mongo.get_pred('ag_news', 'elmo_mean_ln')
        pred = np.array(pred)
        sents = [preprocess_sent(s, stopwords) for s in sents]
        cv = CountVectorizer(max_features=2000, binary=True, stop_words=stopwords)
        feat = cv.fit_transform(sents)
        feat = feat.toarray()
        lg = train_linear_classification(feat, pred)
        feats_list = cv.get_feature_names()
        selected_feat = get_importance_features(lg, feats_list,feat_num=30)
        selected_feat2 = get_importance_features_count(sents, pred, feat_num=30)
        mongo.insert_result('ag_news','elmo_mean_ln',selected_feat)
        mongo.insert_result('ag_news','elmo_mean_ln',selected_feat2,label='count')
        pred = mongo.get_pred('ag_news', 'tfidf',collection_name='other_results')
        pred = np.array(pred)
        lg = train_linear_classification(feat, pred)
        feats_list = cv.get_feature_names()
        selected_feat = get_importance_features(lg, feats_list,feat_num=30)
        selected_feat2 = get_importance_features_count(sents, pred, feat_num=30)
        mongo.insert_result('ag_news','tfidf',selected_feat)
        mongo.insert_result('ag_news','tfidf',selected_feat2,label='count')
        mongo.close()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import csv
import json
import lda
import os

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size, 'y_pred.size {} y_true.size {}'.format(y_pred.size, y_true.size)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

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

def cluster_alg(feat, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=10, verbose=True)
    pred = kmeans.fit_predict(feat)
    return pred

def lda_cluster_alg(feat, n_clusters, n_topics=50):
    model = lda.LDA(n_topics=n_topics, alpha=n_topics/50.0, eta=0.1)
    model.fit(feat)
    pred = np.argmax(model.doc_topic_, axis=1)
    return pred

def lda_kmeans_cluster_alg(feat, n_clusters, n_topics=50):
    model = lda.LDA(n_topics=n_topics, alpha=n_topics/50.0, eta=0.1)
    model.fit(feat)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=10, verbose=True)
    pred = kmeans.fit_predict(model.doc_topic_)
    return pred

def dump_mongo(corpora, feat_name, n_topics, acc, pred, all_pred, all_acc, all_nmi, all_ari):
    acc_std = np.std(all_acc)
    acc_mean = np.mean(all_acc)
    nmi_std = np.std(all_nmi)
    nmi_mean = np.mean(all_nmi)
    ari_std = np.std(all_ari)
    ari_mean = np.mean(all_ari)
    best_nmi = np.max(all_nmi)
    best_ari = np.max(all_ari)
    tmp = {
            'corpora': corpora,
            'feat_name': feat_name,
            'n_topics': n_topics,
            'best_pred': pred,
            'best_acc': acc,
            'best_nmi':best_nmi,
            'best_ari':best_ari,
            'all_pred': all_pred,
            'all_acc': all_acc,
            'acc_std':acc_std,
            'acc_mean':acc_mean,
            'all_nmi':all_nmi,
            'nmi_std':nmi_std,
            'nmi_mean':nmi_mean,
            'all_ari':all_ari,
            'ari_std':ari_std,
            'ari_mean':ari_mean}
    print(tmp)
    with open('tfidf_results.txt','a') as f:
        import json
        f.write(json.dumps(tmp))
        f.write('\n')
    if False:
        from pymongo import MongoClient
        client = MongoClient('59.72.109.90', 27017)
        cluster_db = client.cluster_db
        results = cluster_db.other_results
        results.insert_one(tmp)
        client.close()

# data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers', 3:'reuters'}
# n_cluster_dict = {0: 4, 1: 14, 2: 10, 3:10}
data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers', 3:'reuters_2', 4:'reuters_5', 5:'reuters_10', 6:'reuters_19'}
n_cluster_dict = {0: 4, 1: 14, 2: 10, 3:2, 4:5, 5:10, 6:19}

if __name__ == '__main__':
    from collections import namedtuple
    ARGS= namedtuple('ARGS', ['corpora_id', 'batch_size'])
    for corpora_id in range(3, 7):
        args = ARGS(corpora_id=corpora_id, batch_size=32)
        corpora_name = data_dict[args.corpora_id]
        n_clusters = n_cluster_dict[args.corpora_id]
        train_path = os.path.join('data', corpora_name, 'train.csv')
        # sents, labels = load_json_data(train_path)
        sents, labels, _ = load_csv_corpus(train_path)
        feat = get_tf_idf_feat(sents)
        trial_num = 10

        feat_name = 'tfidf'
        best_acc = 0.0
        best_pred = None
        feat_tmp = feat
        all_pred = []
        all_acc = []
        all_nmi = []
        all_ari = []
        for i in range(trial_num):
            pred = cluster_alg(feat, n_clusters)
            acc = cluster_acc(labels, pred)
            nmi = normalized_mutual_info_score(labels, pred)
            ari = adjusted_mutual_info_score(labels, pred)
            all_pred.append(pred.tolist())
            all_acc.append(acc)
            all_nmi.append(nmi)
            all_ari.append(ari)
            if acc > best_acc:
                best_pred = pred
                best_acc = acc
        print('{} best acc is {}'.format(feat_name, best_acc))
        dump_mongo(corpora=corpora_name, 
                feat_name=feat_name, 
                n_topics=n_clusters,
                pred=best_pred.tolist(),
                acc=best_acc,
                all_pred=all_pred,
                all_acc=all_acc,
                all_nmi=all_nmi,
                all_ari=all_ari)

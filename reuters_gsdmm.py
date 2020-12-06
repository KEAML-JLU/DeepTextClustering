import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import json
import lda
import jpype

jpype.startJVM(jpype.getDefaultJVMPath(), '-Djava.class.path=%s'%('GSDMM.jar',))
GSDMM = jpype.JClass('main.GSDMM')

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
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


def gsdmm_cluster_alg(dataset, n_topics, alpha=0.1, beta=0.1, iter_nums=10):
    gsdmm = GSDMM(n_topics, alpha, beta, iter_nums, dataset)
    pred = gsdmm.gsdmm_cluser()
    return np.array([pred[i] for i in range(len(pred))])

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
    with open('gsdmm_results.txt','a') as f:
        f.write(json.dumps(tmp) + '\n')
    if False:
        from pymongo import MongoClient
        client = MongoClient('59.72.109.90', 27017)
        cluster_db = client.cluster_db
        results = cluster_db.other_results
        results.insert_one(tmp)
        client.close()

# data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers'}
# n_cluster_dict = {0: 4, 1: 14, 2: 10}
data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers', 3:'reuters_2', 4:'reuters_5', 5:'reuters_10', 6:'reuters_19'}
n_cluster_dict = {0: 4, 1: 14, 2: 10, 3:2, 4:5, 5:10, 6:19}

if __name__ == '__main__':
    if False:
        def get_args():
            import argparse
            parser = argparse.ArgumentParser(description='Comparision Experiments')
            parser.add_argument('--corpora_id', type=int, default=0, help='corpora id')
            parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
            parser.add_argument('--beta', type=float, default=0.1, help='beta')
            parser.add_argument('--iter_nums', type=int, default=10, help='iter_nums')
            args = parser.parse_args()
            return args
        args = get_args()
        assert 0 <= args.corpora_id <= 2
    from collections import namedtuple
    ARGS= namedtuple('ARGS', ['corpora_id', 'batch_size'])
    for corpora_id in range(3, 7):
        args = ARGS(corpora_id=corpora_id, batch_size=32)
        corpora_name = data_dict[args.corpora_id]
        n_clusters = n_cluster_dict[args.corpora_id]
        train_path = os.path.join('data', corpora_name, 'data.gsdmm')
        sents, labels = load_json_data(train_path)
        alpha = 0.1
        beta = 0.1
        iter_nums = 10
        trial_num = 10
        max_topic = 30

        for n_topics in range(n_clusters, min((max_topic, n_clusters * 2))):
            if n_topics < n_clusters:
                continue
            feat_name = 'gsdmm'
            best_acc = 0.0
            best_pred = None
            all_pred = []
            all_acc = []
            all_nmi = []
            all_ari = []
            for i in range(trial_num):
                print(corpora_id, n_topics, i)
                # pred = gsdmm_cluster_alg(corpora_name, n_topics, alpha, beta, iter_nums)
                pred = gsdmm_cluster_alg(train_path, n_topics, alpha, beta, iter_nums)
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
                    n_topics=n_topics,
                    pred=best_pred.tolist(),
                    acc=best_acc,
                    all_pred=all_pred,
                    all_acc=all_acc,
                    all_nmi=all_nmi,
                    all_ari=all_ari)


import numpy as np
import csv
import os
from sklearn.cluster import KMeans
from utils import dump_feat, load_feat
import torch
from config import cfg
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from config import cfg
from sklearn.decomposition import PCA


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


def pca_feat_transform(feat, hidden_dim=100):
    print('pca')
    pca = PCA(n_components=hidden_dim)
    t_feat = pca.fit_transform(feat)
    return t_feat

def cluster_alg(feat, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=10, verbose=True)
    pred = kmeans.fit_predict(feat)
    return pred

def ln(feat):
    return (feat - feat.mean(axis=1, keepdims=True)) / feat.std(axis=1, keepdims=True)

def norm(feat):
    return feat / np.linalg.norm(feat, axis=1, keepdims=True)

def dump_mongo(corpora, feat_name, n_topics, acc, pred, all_pred, all_acc, all_nmi, all_ari,n_components):
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
            'ari_mean':ari_mean,
            'n_components':n_components}
    print(tmp)
    with open('pca_results.txt','a') as f:
        import json
        f.write(json.dumps(tmp))
        f.write('\n')
    if False:
        from pymongo import MongoClient
        client = MongoClient('59.72.109.90', 27017)
        cluster_db = client.cluster_db
        results = cluster_db.pca_results
        results.insert_one(tmp)
        client.close()

# data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers'}
data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers', 3:'reuters_2', 4:'reuters_5', 5:'reuters_10', 6:'reuters_19'}
feat_dict = {0:'infersent',1:'elmo_max', 2:'elmo_mean', 3:'tfidf'}
feat_func_dict = {'ln': ln, 'n': norm, 'i': lambda x: x}
# n_cluster_dict = {0: 4, 1: 14, 2: 10}
n_cluster_dict = {0: 4, 1: 14, 2: 10, 3:2, 4:5, 5:10, 6:19}
input_feat_size_dict = {0: 4096,1:1024,2:1024, 3:2000}

if __name__ == '__main__':
    if False:
        def get_args():
            import argparse
            parser = argparse.ArgumentParser(description='ElMo')
            parser.add_argument('--corpora_id', type=int, default=0, help='corpora id')
            parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
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

        root_dir = os.path.join('data', corpora_name)
        for feat_id, feat_name in feat_dict.items():
            train_feat_path = os.path.join(root_dir, feat_name+'.h5')
            raw_train_feat, labels, _  = load_feat(train_feat_path)
            for feat_func_name, feat_func in feat_func_dict.items():
                t_raw_train_feat = feat_func(raw_train_feat)
                for n_components in [50, 100, 200, 300]:
                    try:
                        feat = pca_feat_transform(t_raw_train_feat, hidden_dim=n_components)
                    except:
                        continue
                    trial_num = 10
                    best_acc = 0.0
                    best_pred = None
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
                    if len(all_acc) == 0:
                        continue
                    tmp_feat_name = feat_name + '_{}'.format(feat_func_name) + '_pca_{}'.format(n_components)
                    print('{} {} {} best acc is {}'.format(tmp_feat_name, feat_func_name, n_components, best_acc))
                    pred_std = np.std(all_acc)
                    pred_mean = np.mean(all_acc)
                    dump_mongo(corpora=corpora_name, 
                            feat_name=tmp_feat_name, 
                            n_topics=n_clusters, 
                            acc=best_acc, 
                            pred=best_pred.tolist(), 
                            all_pred=all_pred, 
                            all_acc=all_acc, 
                            all_nmi=all_nmi, 
                            all_ari=all_ari,
                            n_components=n_components)

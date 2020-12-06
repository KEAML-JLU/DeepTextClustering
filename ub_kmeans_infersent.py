import numpy as np
import csv
import os
from sklearn.cluster import KMeans
from utils import load_infersent, load_csv_corpus, infersent_encode_sents, dump_feat
import torch
from config import cfg
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score


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


def cluster_alg(feat, n_clusters, n_jobs=3):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=n_jobs, verbose=True)
    pred = kmeans.fit_predict(feat)
    return pred

def ln(feat):
    return (feat - feat.mean(axis=1, keepdims=True)) / feat.std(axis=1, keepdims=True)

def norm(feat):
    return feat / np.linalg.norm(feat, axis=1, keepdims=True)

def dump_mongo(corpora, feat_name, n_topics, acc, pred, all_pred, all_acc, all_nmi, all_ari):
    from pymongo import MongoClient
    client = MongoClient('59.72.109.90', 27017)
    cluster_db = client.cluster_db
    results = cluster_db.ub_results
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
    results.insert_one(tmp)
    client.close()

feat_func_dict = {'ln': ln, 'n': norm, 'i': lambda x: x}
data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers'}
n_cluster_dict = {0: 4, 1: 14, 2: 10}

if __name__ == '__main__':
    def get_args():
        import argparse
        parser = argparse.ArgumentParser(description='ElMo')
        parser.add_argument('--corpora_id', type=int, default=0, help='corpora id')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        args = parser.parse_args()
        return args
    args = get_args()
    assert 0 <= args.corpora_id <= 2
    corpora_name = data_dict[args.corpora_id]
    n_clusters = n_cluster_dict[args.corpora_id]
    train_path = os.path.join('data', corpora_name, 'ub_train.csv')
    #
    print('Loading Pretrained Infersent Model')
    infersent = load_infersent(cfg.INFERSENT_PATH, return_adaptor=True, use_cuda=torch.cuda.is_available())
    infersent.set_glove_path(cfg.GLOVE_PATH)
    #
    
    train_feat, train_labels, train_ids = get_feat(infersent, train_path, verbose=True)

    #
    trial_num = 10
    #

    feat = train_feat
    labels = train_labels
    feat_name='Infersent'
    for func_name, feat_trans_func in feat_func_dict.items():
        best_acc = 0.0
        best_pred = None
        feat_tmp = feat_trans_func(feat)
        all_pred = []
        all_acc = []
        all_nmi = []
        all_ari = []
        for i in range(trial_num):
            pred = cluster_alg(feat_tmp, n_clusters)
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
        tmp_feat_name = feat_name + '_{}'.format(func_name)
        print('{} {} best acc is {}'.format(tmp_feat_name, func_name, best_acc))
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
                all_ari=all_ari)

from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import csv
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

encoder = ElmoEmbedder(
        options_file='elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json',
        weight_file='elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
        cuda_device=0)

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

def feat_extraction(sents, batch_size=32):
    data_size = len(sents)
    feat_max = []
    feat_mean = []
    feat_last = []
    for i in range(0, data_size, batch_size):
        batch_sents = sents[i: i + batch_size]
        batch_feat_lst = encoder.embed_batch([s.split() for s in batch_sents])
        feat_max.extend([np.max(tmp[-1], axis=0) for tmp in batch_feat_lst])
        feat_mean.extend([np.mean(tmp[-1], axis=0) for tmp in batch_feat_lst])
        feat_last.extend([tmp[-1][-1] for tmp in batch_feat_lst])
        print(i)
    return np.stack(feat_max), np.stack(feat_mean), np.stack(feat_last)

def cluster_alg(feat, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=10, verbose=True)
    pred = kmeans.fit_predict(feat)
    return pred

def ln(feat):
    return (feat - feat.mean(axis=1, keepdims=True)) / feat.std(axis=1, keepdims=True)

def norm(feat):
    return feat / np.linalg.norm(feat, axis=1, keepdims=True)

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
    with open('elmo_results.txt','a') as f:
        import json
        f.write(json.dumps(tmp))
        f.write('\n')

    if False:
        from pymongo import MongoClient
        client = MongoClient('59.72.109.90', 27017)
        cluster_db = client.cluster_db
        results = cluster_db.elmo_results
        results.insert_one(tmp)
        client.close()

feat_func_dict = {'ln': ln, 'n': norm, 'i': lambda x: x}
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
        sents, labels, _ = load_csv_corpus(train_path)
        labels = np.array(labels)
        feat_max, feat_mean, feat_last = feat_extraction(sents, batch_size=args.batch_size)
        all_feat = {'elmo_max':feat_max, 'elmo_mean':feat_mean, 'elmo_last': feat_last}
        #
        trial_num = 10
        #

        for feat_name, feat in all_feat.items():
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

import csv
import os
import random

import nltk
import torch
import numpy as np
from torch.backends import cudnn as cudnn

from models import InfersentAdaptor


def load_infersent(infersent_path, return_adaptor=False, use_cuda=torch.cuda.is_available()):
    infersent = torch.load(infersent_path) if use_cuda else \
        torch.load(infersent_path, map_location=lambda storage, loc: storage)
    if return_adaptor:
        infersent = InfersentAdaptor(infersent)
    return infersent


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


def encode_sents(infersent, sents, split_sents=False, layer_norm=False):
    if False:
        if split_sents:
            all_feat = []
            for doc in sents:
                tmp_sents = nltk.sent_tokenize(doc)
                tmp_feat = infersent.encode(tmp_sents, tokenize=False, layer_norm=layer_norm)
                all_feat.append(tmp_feat.mean(axis=0))
            feat = np.stack(all_feat, axis=0)
        else:
            feat = infersent.encode(sents, tokenize=False, layer_norm=layer_norm)
    else:
        feat = infersent_encode_sents(infersent, sents, split_sents=split_sents, layer_norm=layer_norm)
    return feat


def infersent_encode_sents(infersent, sents, split_sents=False, layer_norm=False, batch_size=256, verbose=False):
    if split_sents:
        all_feat = []
        batch_sents = []
        batch_slens = []
        for doc_id, doc in enumerate(sents):
            tmp_sents = nltk.sent_tokenize(doc)
            # Put current batch and batch_size respectively into batch_sents and batch_slens.
            batch_sents.extend(tmp_sents)
            batch_slens.append(len(batch_sents))
            # If current sents list size > batch_size,
            # encode all sents by infersent, get paragraph vector and reset batch_sents and batch_slens.
            if len(batch_sents) >= batch_size or doc_id == len(sents) - 1:
                batch_feat = infersent.encode(batch_sents, tokenize=False, layer_norm=layer_norm)
                for i, tmp_len in enumerate(batch_slens):
                    bidx = batch_slens[i-1] if i > 0 else 0
                    eidx = batch_slens[i]
                    tmp_feat = batch_feat[bidx:eidx]
                    all_feat.append(tmp_feat.mean(axis=0))
                batch_sents = []
                batch_slens = []
                if verbose:
                    print('Infersent Processed {} Text'.format(doc_id+1))
        assert len(sents) == len(all_feat)
        feat = np.stack(all_feat, axis=0)
    else:
        feat = infersent.encode(sents, tokenize=False, layer_norm=layer_norm)
    return feat


def load_constraint_file(path):
    constraints = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for item in reader:
            constraints.append((int(item[0]), int(item[1])));
    return constraints


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


def align_labels(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size, 'y_pred.size {} y_true.size {}'.format(y_pred.size, y_true.size)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return ind


def dump_feat(feat_path, feat, labels=None, ids=None):
    import h5py
    f = h5py.File(feat_path, 'w')
    f['feat'] = feat.astype(dtype=np.float32)
    if labels is not None:
        f['labels'] = np.array(labels)
    if ids is not None:
        f['ids'] = np.array(ids)
    f.close()


def load_feat(feat_path):
    import h5py
    f = h5py.File(feat_path, 'r')
    feat = np.array(f['feat'], dtype=np.float32)
    labels = None
    ids = None
    if 'labels' in f.keys():
        labels = np.array(f['labels'])
    if 'ids' in f.keys():
        ids = np.array(f['ids'])
    return feat, labels, ids


def initialize_environment(random_seed=50, use_cuda=torch.cuda.is_available()):
    # Set the seed for reproducing the results
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True


def load_seeds_dict(path):
    import csv
    from collections import defaultdict
    results = defaultdict(list)
    if os.path.exists(path):
        with open(path) as f:
            reader = csv.reader(f)
            for item in reader:
                i = int(item[0])
                l = int(item[1])
                results[l].append(i)
    return results

import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data as data

from config import cfg
from utils import load_constraint_file, load_feat


class EncodedTextDataset(data.Dataset):
    """Custom dataset loader for Pretraining SDAE"""
    def __init__(self, root, train=True, verbose=True):
        self.root_dir = root
        self.train = train
        self.verbose = verbose

        if self.train:
            train_feat_path = os.path.join(self.root_dir, cfg.TRAIN_TEXT_FEAT_FILE_NAME)
            self.train_data, self.train_labels, _ = load_feat(train_feat_path)
            self.train_ids = np.array(range(len(self.train_labels)))
            if self.verbose:
                print('Loading {} training item'.format(len(self.train_labels)))
        else:
            test_feat_path = os.path.join(self.root_dir, cfg.TEST_TEXT_FEAT_FILE_NAME)
            self.test_data, self.test_labels, _ = load_feat(test_feat_path)
            self.test_ids = np.array(range(len(self.test_labels)))
            if self.verbose:
                print('Loading {} testing items'.format(len(self.test_labels)))

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, item):
        if self.train:
            data, target, id = self.train_data[item], self.train_labels[item], self.train_ids[item]
        else:
            data, target, id = self.test_data[item], self.test_labels[item], self.test_ids[item]
        return data, target, id


class Corpus_Loader(object):
    def __init__(self, root_dir, verbose=True, layer_norm=True, split_sents=False, semi_supervised=False, use_cuda=torch.cuda.is_available()):
        self.root_dir = root_dir
        self.verbose = verbose
        self.use_cuda = use_cuda
        self.semi_supervised = semi_supervised

        if self.semi_supervised:
            constraint_file_path = os.path.join(self.root_dir, cfg.CONSTRAINTS_NAME + '.csv')
            self.constraints = load_constraint_file(constraint_file_path)

        train_feat_path = os.path.join(self.root_dir, cfg.TRAIN_TEXT_FEAT_FILE_NAME)
        self.train_fixed_features, self.train_labels, _ = load_feat(train_feat_path)
        self.train_ids = np.array(range(len(self.train_labels)))
        if self.verbose:
            print('Loading {} training item'.format(len(self.train_labels)))

    @property
    def data_size(self):
        return len(self.train_labels)

    def train_data_iter(self, batch_size, shuffle=False, infinite=False):
        while True:
            return_idx = np.arange(self.data_size)
            if shuffle:
                np.random.shuffle(return_idx)
            for i in range(0, self.data_size, batch_size):
                id_batch = np.array([self.train_ids[j] for j in return_idx[i:i + batch_size]])
                label_batch = np.array([self.train_labels[j] for j in return_idx[i:i + batch_size]])
                fixed_feat_batch = self.train_fixed_features[return_idx[i:i + batch_size]]
                fixed_feat_batch = Variable(torch.Tensor(fixed_feat_batch))
                if self.use_cuda:
                    fixed_feat_batch = fixed_feat_batch.cuda()
                # fixed_feat_batch is Variable of torch.Tensor
                # sent_feat_batch is Variable of torch.Tensor
                # id_batch is numpy.darray
                # label_batch is numpy.darray
                yield fixed_feat_batch, label_batch, id_batch
            # If infinite is set to True, for-loop will infinitely iterate
            # If infinite is set to False,  for-loop will iterate one time.
            if not infinite:
                break

    def constraint_data_iter(self, batch_size, shuffle=False, infinite=False):
        if not self.semi_supervised:
            raise RuntimeError('current mode is unsupervised')
        while True:
            constraints_size = len(self.constraints)
            return_idx = np.arange(constraints_size)
            if shuffle:
                np.random.shuffle(return_idx)
            for i in range(0, constraints_size, batch_size):
                constraints_batch = [self.constraints[j] for j in return_idx[i:i + batch_size]]
                cons_id1 = np.array([item[0] for item in constraints_batch])
                cons_id2 = np.array([item[1] for item in constraints_batch])
                feat_batch1 = self.train_fixed_features[cons_id1]
                feat_batch2 = self.train_fixed_features[cons_id2]
                feat_batch1 = Variable(torch.Tensor(feat_batch1))
                feat_batch2 = Variable(torch.Tensor(feat_batch2))
                if self.use_cuda:
                    feat_batch1 = feat_batch1.cuda()
                    feat_batch2 = feat_batch2.cuda()
                yield feat_batch1, feat_batch2
            if not infinite:
                break

    def get_fixed_features(self):
        return self.train_fixed_features

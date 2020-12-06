import argparse
import os
import torch
import numpy as np

from config import cfg
from utils import load_infersent, load_csv_corpus, infersent_encode_sents, dump_feat


def parse_args():
    parser = argparse.ArgumentParser(description='Building Text Representation')
    parser.add_argument('--data_dir', dest='db_dir', type=str, default='data/ag_news', help='directory of dataset')
    parser.add_argument('--model_id', type=int, default=0,
                        help='feature extractor model\'s id (0:Infersent, 1:Tfidf2000, 2:Tfidf5000)')
    parser.add_argument('--split_sents', help='whether split sents before learning', action='store_true')
    parser.add_argument('--layer_norm', help='whether use layer norm', action='store_true')
    parser.add_argument('--dump_ids', help='whether use layer norm', action='store_true')
    args = parser.parse_args()
    return args


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


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.db_dir
    model_id = args.model_id
    assert 0 <= model_id <= 2
    split_sents = args.split_sents
    layer_norm = args.layer_norm
    dump_ids = args.dump_ids

    train_data_path = os.path.join(data_dir, cfg.TRAIN_DATA_NAME+'.csv')
    test_data_path = os.path.join(data_dir, cfg.TEST_DATA_NAME+'.csv')
    train_feat_path = os.path.join(data_dir, cfg.TRAIN_TEXT_FEAT_FILE_NAME)
    test_feat_path = os.path.join(data_dir, cfg.TEST_TEXT_FEAT_FILE_NAME)

    print('Loading Pretrained Infersent Model')
    infersent = load_infersent(cfg.INFERSENT_PATH, return_adaptor=True, use_cuda=torch.cuda.is_available())
    infersent.set_glove_path(cfg.GLOVE_PATH)

    print('Building feat for {}'.format(train_data_path))
    train_feat, train_labels, train_ids = get_feat(infersent, train_data_path, verbose=True, layer_norm=layer_norm, split_sents=split_sents)
    print('Dumping Train Text Feat and Labels into {}'.format(train_feat_path))
    dump_feat(train_feat_path, train_feat, labels=train_labels)

    if os.path.exists(test_data_path):
        print('Building feat for {}'.format(test_data_path))
        test_feat, test_labels, test_ids = get_feat(infersent, test_data_path, verbose=True, layer_norm=layer_norm, split_sents=split_sents)
        print('Dumping Test Text Feat and Labels into {}'.format(test_feat_path))
        dump_feat(test_feat_path, test_feat, labels=test_labels)


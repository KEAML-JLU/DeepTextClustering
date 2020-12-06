from allennlp.commands.elmo import ElmoEmbedder
import os
import csv
import numpy as np
# encoder = ElmoEmbedder(cuda_device=0)
encoder = ElmoEmbedder(
        options_file='elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json',
        weight_file='elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
        cuda_device=0)

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

def dump_feat(feat_path, feat, labels=None, ids=None):
    import h5py
    f = h5py.File(feat_path, 'w')
    f['feat'] = feat.astype(dtype=np.float32)
    if labels is not None:
        f['labels'] = np.array(labels)
    if ids is not None:
        f['ids'] = np.array(ids)
    f.close()

# data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers'}
data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers', 3:'reuters_2', 4:'reuters_5', 5:'reuters_10', 6:'reuters_19'}
if __name__ == '__main__':
    def get_args():
        import argparse
        parser = argparse.ArgumentParser(description='ElMo')
        parser.add_argument('--corpora_id', type=int, default=0, help='corpora id')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        args = parser.parse_args()
        return args
    args = get_args()
    assert 0 <= args.corpora_id <= 6
    corpora_name = data_dict[args.corpora_id]
    train_path = os.path.join('data', corpora_name, 'train.csv')
    max_feat_path = os.path.join('data',corpora_name, 'elmo_max.h5')
    mean_feat_path = os.path.join('data', corpora_name, 'elmo_mean.h5')
    sents, labels, _ = load_csv_corpus(train_path)
    feat_max, feat_mean, _ = feat_extraction(sents, batch_size=args.batch_size)
    dump_feat(max_feat_path, feat_max, labels)
    dump_feat(mean_feat_path, feat_mean, labels)

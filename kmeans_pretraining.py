import os
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn.cluster import KMeans

from SDAE import extract_sdae_model
from utils import load_feat
from config import get_output_dir, cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Fine Tuning NN by KMeans')
    parser.add_argument('--data_dir', dest='db_dir', type=str, default='data/ag_news', help='directory of dataset')
    parser.add_argument('--n_clusters', dest='n_clusters', type=int, default=14, help='cluster num')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--nepochs', dest='nepochs', type=int, default=3, help='epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='epoch')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.1, help='epoch')
    parser.add_argument('--print_every', dest='print_every', type=int, default=10, help='epoch')
    args = parser.parse_args()
    return args


def feat_cluster(feat, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=10)
    pred = kmeans.fit_predict(feat)
    return pred


def load_pretrained_fd_autoencoder(net_filename, use_cuda=torch.cuda.is_available()):
    """
    load pretrained stack denoise autoencoder
    """
    checkpoint = torch.load(net_filename)
    fd_ae = extract_sdae_model(cfg.INPUT_DIM, cfg.HIDDEN_DIMS)
    # there some problems when loading cuda pretrained models
    fd_ae.load_state_dict(checkpoint['state_dict'])
    if use_cuda:
        fd_ae.cuda()
    return fd_ae


def dump_fd_autoencoder(net_filename, net):
    torch.save({'state_dict': net.state_dict()}, net_filename)


args = parse_args()
outputdir = get_output_dir(args.db_dir)
net_filename = os.path.join(outputdir, cfg.PRETRAINED_FAE_FILENAME)
feat_filename = os.path.join(args.db_dir, cfg.TRAIN_TEXT_FEAT_FILE_NAME)
feat, _,_ = load_feat(feat_filename)
data_size = feat.shape[0]
batch_size = args.batch_size
nepochs = args.nepochs
n_clusters = args.n_clusters
use_cuda = torch.cuda.is_available()
lr = args.lr
gamma = args.gamma
print_every = args.print_every

print('Get pseudo labels by KMeans')
pseu_labels = feat_cluster(feat, n_clusters=n_clusters)

print('Get pretrained SDAE')
ae_net = load_pretrained_fd_autoencoder(net_filename, use_cuda=use_cuda)
classifier = torch.nn.Linear(cfg.HIDDEN_DIMS[-1], n_clusters)
if use_cuda:
    classifier.cuda()

optimizer = optim.Adam([{'params':ae_net.parameters()}, {'params': classifier.parameters()}], lr=lr)
recons_criteron = torch.nn.MSELoss(size_average=True)
class_criteron = torch.nn.CrossEntropyLoss(size_average=True)

for epoch in range(nepochs):
    for i in range(0, data_size, batch_size):
        feat_batch = Variable(torch.from_numpy(feat[i: i+batch_size]))
        labels_batch = Variable(torch.from_numpy(pseu_labels[i: i+batch_size]))
        if use_cuda:
            feat_batch, labels_batch = feat_batch.cuda(), labels_batch.cuda()
        optimizer.zero_grad()
        hidden_feat, output_feat = ae_net(feat_batch)
        pred_score_batch = classifier(hidden_feat)
        class_loss = class_criteron(pred_score_batch, labels_batch)
        recons_loss = recons_criteron(output_feat, feat_batch)
        loss = gamma * class_loss + recons_loss
        loss.backward()
        optimizer.step()
        if i % print_every == 0:
            iter = int(i / batch_size)
            class_loss = class_loss.cpu().data[0]
            recons_loss = recons_loss.cpu().data[0]
            pred_score_batch = pred_score_batch.cpu().data
            pred_labels = torch.max(pred_score_batch, dim=1)[0]
            acc = torch.sum(pred_labels == labels_batch.cpu().data) / float(labels_batch.size(0))
            print('Epoch {} Iter {} class_loss {} recons_loss {} acc {}'.format(epoch+1, i+1, class_loss, recons_loss, acc))
    tmp_correct = 0
    for i in range(0, data_size, batch_size):
        feat_batch = Variable(torch.from_numpy(feat[i: i+batch_size]))
        labels_batch = Variable(torch.from_numpy(pseu_labels[i: i+batch_size]))
        if use_cuda:
            feat_batch, labels_batch = feat_batch.cuda(), labels_batch.cuda()
        hidden_feat, _ = ae_net(feat_batch)
        pred_score_batch = classifier(hidden_feat)
        pred_score_batch = pred_score_batch.cpu().data
        pred_labels = torch.max(pred_score_batch, dim=1)[0]
        tmp_correct += torch.sum(pred_labels == labels_batch.cpu().data)
    print('Epoch {} end acc is {}'.format(epoch+1, tmp_correct / float(data_size)))

print('Dumping AE')
dump_fd_autoencoder(net_filename, ae_net)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.cluster import KMeans
from utils import cluster_acc


class DCN(object):
    def __init__(self,
                 n_clusters,
                 net,
                 hidden_dim,
                 lr=0.001,
                 tol=0.001,
                 batch_size=256,
                 max_epochs=100,
                 recons_lam=1,
                 cluster_lam=0.5,
                 use_cuda=torch.cuda.is_available(),
                 verbose=True):
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.tol = tol
        self.max_epochs = max_epochs
        self.recons_lam = recons_lam
        self.cluster_lam = cluster_lam
        self.use_cuda = use_cuda
        self.verbose = verbose
        self.net = net
        assert isinstance(self.net, nn.Module)
        self.centers = None

    def fit(self, feat, labels=None):
        feat = feat.astype(np.float32)
        batch_size = self.batch_size
        data_size = feat.shape[0]
        count = {i: 0 for i in range(self.n_clusters)}

        hidden_feat = self.get_hidden_features(feat, self.net, self.hidden_dim, batch_size=self.batch_size, use_cuda=self.use_cuda)
        idx, centers = self.init_cluster(hidden_feat, n_clusters=self.n_clusters)
        last_pred = idx[:]
        if labels is not None:
            acc = cluster_acc(labels, idx)
            print('KMeans pretraining acc is {}'.format(acc))

        # optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # optimizer = optim.ASGD(self.net.parameters(), lr=self.lr)
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        for epoch in range(self.max_epochs):
            if False:
                if epoch < 10:
                    count = {i: 50 for i in range(self.n_clusters)}
            for index in range(0, data_size, batch_size):
                feat_batch = Variable(torch.from_numpy(feat[index: index+batch_size]))
                idx_batch = idx[index: index+batch_size]
                centers_batch = Variable(torch.from_numpy(centers[idx_batch]))
                if self.use_cuda:
                    feat_batch = feat_batch.cuda()
                    centers_batch = centers_batch.cuda()
                optimizer.zero_grad()
                hidden_batch, output_batch = self.net(feat_batch)
                recons_loss = F.mse_loss(output_batch, feat_batch)
                cluster_loss = F.mse_loss(hidden_batch, centers_batch)
                loss = self.recons_lam * recons_loss + self.cluster_lam * cluster_loss
                loss.backward()
                optimizer.step()
                hidden_batch2, _ = self.net(feat_batch)
                hidden_batch2 = hidden_batch2.cpu().data.numpy()
                tmp_idx_batch, centers, count = self.batch_km(hidden_batch2, centers, count)
                idx[index: index+batch_size] = tmp_idx_batch

            hidden_feat = self.get_hidden_features(feat, self.net, self.hidden_dim, batch_size=self.batch_size, use_cuda=self.use_cuda)
            idx, centers = self.init_cluster(hidden_feat, n_clusters=self.n_clusters, init_centers=centers)
            acc = None
            if labels is not None:
                acc = cluster_acc(labels, idx)
            if self.verbose:
                print('Epoch {} end, current acc is {}'.format(epoch + 1, acc))
            if self.whether_convergence(last_pred, idx, self.tol):
                print('End Iter')
                break
            else:
                last_pred = idx[:]
        self.centenrs = centers

    def predict(self, feat):
        hidden_feat = self.get_hidden_features(feat, self.net, self.hidden_dim, batch_size=self.batch_size, use_cuda=self.use_cuda)
        distances = np.linalg.norm(hidden_feat[:,np.newaxis] - self.centers[np.newaxis, :], axis=-1)
        pred = np.argmin(distances, axis=-1)
        return pred

    @staticmethod
    def get_hidden_features(feat, net, hidden_dim, batch_size=256, use_cuda=torch.cuda.is_available()):
        feat = feat.astype(np.float32)
        data_size = feat.shape[0]
        hidden_feat = np.zeros((data_size, hidden_dim))
        for index in range(0, data_size, batch_size):
            data_batch = feat[index: index + batch_size]
            data_batch = Variable(torch.from_numpy(data_batch))
            if use_cuda:
                data_batch = data_batch.cuda()
            hidden_batch, _ = net(data_batch)
            hidden_batch = hidden_batch.data.cpu().numpy()
            hidden_feat[index: index+batch_size] = hidden_batch
        return hidden_feat

    @staticmethod
    def init_cluster(feat, n_clusters, init_centers=None):
        init_centers = 'k-means++' if init_centers is None else init_centers
        kmeans = KMeans(n_clusters=n_clusters, init=init_centers, n_init=20)
        idx = kmeans.fit_predict(feat)
        centers = kmeans.cluster_centers_
        centers = centers.astype(np.float32)
        return idx, centers

    @staticmethod
    def batch_km(data, centers, count):
        # data[:, np.newaxis] is a data_size * 1 * feat_size array
        # centers[np.newaxis, :] is a 1 * center_size * feat_size array
        distances = np.linalg.norm(data[:, np.newaxis] - centers[np.newaxis, :], axis=-1)
        tmp_idx = np.argmin(distances, axis=-1)
        N = tmp_idx.shape[0]
        for i in range(N):
            c = tmp_idx[i]
            count[c] += 1
            eta = 1. / count[c]
            centers[c] = (1 - eta) * centers[c] + eta * data[c]
        return tmp_idx, centers, count

    @staticmethod
    def whether_convergence(last_pred, current_pred, tol):
        delta = np.sum(last_pred != current_pred) / float(len(current_pred))
        return delta < tol

if __name__ == '__main__':
    from utils import load_feat, initialize_environment
    from SDAE import extract_sdae_model
    from config import cfg, get_output_dir
    import os

    def get_args():
        import argparse
        parser = argparse.ArgumentParser(description='Deep Text Cluster Model')
        parser.add_argument('--data_dir', type=str, default='data/dbpedia/', help='directory of dataset')
        parser.add_argument('--n_clusters', type=int, default=14, help='cluster number')
        parser.add_argument('--seed', type=int, default=cfg.RNG_SEED, help='random seed')
        parser.add_argument('--tol', type=float, default=0.001, help='tolerance')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--recons_lam', type=float, default=1, help='reconstruction loss regularization coefficient')
        parser.add_argument('--cluster_lam', type=float, default=0.5, help='cluster loss regularization coefficient')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--max_epochs', type=int, default=100, help='max epochs')
        parser.add_argument('--verbose', help='whether to print log', action='store_true')
        args = parser.parse_args()
        return args

    args = get_args()
    # n_clusters = 4
    # data_dir = 'data/ag_news/'
    data_dir = args.data_dir
    n_clusters = args.n_clusters
    use_cuda = torch.cuda.is_available()
    random_seed = args.seed
    recons_lam = args.recons_lam
    cluster_lam = args.cluster_lam
    batch_size = args.batch_size
    tol = args.tol
    lr = args.lr

    initialize_environment(random_seed=random_seed, use_cuda=use_cuda)

    feat_path = os.path.join(data_dir, cfg.TRAIN_TEXT_FEAT_FILE_NAME)
    feat, labels, ids = load_feat(feat_path)
    outputdir = get_output_dir(data_dir)
    net_filename = os.path.join(outputdir, cfg.PRETRAINED_FAE_FILENAME)
    checkpoint = torch.load(net_filename)
    net = extract_sdae_model(input_dim=cfg.INPUT_DIM, hidden_dims=cfg.HIDDEN_DIMS)
    net.load_state_dict(checkpoint['state_dict'])
    if use_cuda:
        net.cuda()

    dcn = DCN(n_clusters,
              net,
              cfg.HIDDEN_DIMS[-1],
              lr=lr,
              tol=tol,
              batch_size=batch_size,
              recons_lam=recons_lam,
              cluster_lam=cluster_lam,
              use_cuda=use_cuda,
              verbose=True)
    dcn.fit(feat, labels=labels)

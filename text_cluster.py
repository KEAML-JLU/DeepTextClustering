import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import tensorboard_logger
from models import ClusterNet
from utils import cluster_acc


class EKMLogger(object):

    def record_acc(self, acc, step):
        self.logger_value('cluster_acc', acc, step)

    def record_loss(self, loss, step):
        self.logger_value('cluster_loss', loss, step)

    def logger_value(self, field_name, value, step):
        raise NotImplementedError()


class EKM_Tensorboard_Logger(EKMLogger):
    def __init__(self, path):
        super(EKM_Tensorboard_Logger, self).__init__()
        self.logger = tensorboard_logger.Logger(path)

    def logger_value(self, field_name, value, step):
        self.logger.log_value(field_name, value, step)


class EnhancedKMeans(object):
    def __init__(self,
                 n_clusters=4,
                 update_interval=2,
                 tol=0.001,
                 lr=0.001,
                 maxiter=2e4,
                 batch_size=64,
                 max_jobs=10,
                 use_cuda=torch.cuda.is_available(),
                 logger=None,
                 verbose=False):
        self.n_clusters = n_clusters
        self.feat_dim = None
        self.data_size = None
        self.update_interval = update_interval
        self.tol = tol
        self.lr = lr
        self.maxiter = maxiter
        self.batch_size = batch_size
        self.max_jobs = max_jobs
        self.use_cuda = use_cuda
        self.verbose = verbose
        self.logger = logger
        if logger is not None:
            assert isinstance(self.logger, EKMLogger)
        self.kmeans = None
        self.cluster_layer = None
        self.optimizer = None
        self.last_pred = None
        self.current_p = None
        self.current_q = None

    def __initialize_models(self, feat, labels=None):
        self.data_size = feat.shape[0]
        self.feat_dim = feat.shape[1]
        if self.verbose:
            print('Pretraining Cluster Centers by KMeans')
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             n_init=20,
                             n_jobs=self.max_jobs,
                             verbose=False)
        self.last_pred = self.kmeans.fit_predict(feat)

        if labels is not None:
            tmp_acc = cluster_acc(labels, self.last_pred)
            if self.verbose:
                print('KMeans acc is {}'.format(tmp_acc))

        if self.verbose:
            print('Building Cluster Layer')
        # self.cluster_layer = ClusterNet(torch.Tensor(self.kmeans.cluster_centers_.astype(np.float32)))
        self.cluster_layer = ClusterNet(torch.from_numpy(self.kmeans.cluster_centers_.astype(np.float32)))
        if self.use_cuda:
            self.cluster_layer.cuda()
        if self.verbose:
            print('Building Optimizer')
        self.optimizer = optim.Adam(self.cluster_layer.parameters(), lr=self.lr)
        # self.optimizer = optim.SGD(self.cluster_layer.parameters(), lr=self.lr)

    def __update_target_distribute(self, feat):
        if self.verbose:
            print('Updating Target Distribution')
        all_q = np.zeros((self.data_size, self.n_clusters))
        tmp_size = 0
        for i in range(0, self.data_size, self.batch_size):
            tmp_feat = feat[i:i+self.batch_size].astype(np.float32)
            tmp_feat = Variable(torch.from_numpy(tmp_feat))
            if self.use_cuda:
                tmp_feat = tmp_feat.cuda()
            q_batch = self.cluster_layer(tmp_feat)
            q_batch = q_batch.cpu().data.numpy()
            all_q[i:i+self.batch_size] = q_batch
            tmp_size += len(q_batch)
        assert tmp_size == self.data_size
        self.current_q = all_q
        self.current_p = self.__get_target_distribution(self.current_q)

    @staticmethod
    def __get_target_distribution(q):
        p = np.power(q, 2) / np.sum(q, axis=0, keepdims=True)
        p = p / np.sum(p, axis=1, keepdims=True)
        return p

    @staticmethod
    def __get_label_pred(q):
        pred = np.argmax(q, axis=1)
        return pred

    def __whether_convergence(self, pred_cur, pred_last):
        delta_label = np.sum(pred_cur != pred_last) / float(len(pred_cur))
        return delta_label < self.tol

    def fit(self, feat, labels=None):
        self.__initialize_models(feat, labels=labels)
        self.__update_target_distribute(feat)

        if self.verbose:
            print('Begin to Iterate')
        index = 0
        for ite in range(int(self.maxiter)):
            if ite % self.update_interval == (self.update_interval - 1):
                self.__update_target_distribute(feat)
                tmp_pred_cur = self.__get_label_pred(self.current_q)
                acc = None
                if labels is not None:
                    acc = cluster_acc(labels, tmp_pred_cur)
                    if self.logger is not None:
                        self.logger.record_acc(acc, ite)
                if self.verbose:
                    if acc is not None:
                        print('Iter {} Acc {}'.format(ite,acc))
                    else:
                        print('Update Target Distribution in Iter {}'.format(ite))

                if ite > 0 and self.__whether_convergence(tmp_pred_cur, self.last_pred):
                    break
                self.last_pred = tmp_pred_cur

            if index + self.batch_size > self.data_size:
                feat_batch = feat[index:]
                p_batch = self.current_p[index:]
                index = 0
            else:
                feat_batch = feat[index: index + self.batch_size]
                p_batch = self.current_p[index: index + self.batch_size]
            feat_batch = Variable(torch.from_numpy(feat_batch.astype(np.float32)))
            p_batch = Variable(torch.from_numpy(p_batch.astype(np.float32)))
            if self.use_cuda:
                feat_batch = feat_batch.cuda()
                p_batch = p_batch.cuda()

            self.cluster_layer.zero_grad()
            q_batch = self.cluster_layer(feat_batch)
            cluster_loss = F.binary_cross_entropy(q_batch, p_batch)
            if self.logger is not None:
                self.logger.record_loss(cluster_loss.data[0], ite)
            cluster_loss.backward()
            self.optimizer.step()

if __name__ == '__main__':
    from config import cfg
    import h5py
    import os
    root_dir = 'data/dbpedia/'
    n_clusters = 14
    text_feat_path = os.path.join(root_dir, cfg.TRAIN_TEXT_FEAT_FILE_NAME)
    f = h5py.File(text_feat_path, 'r')
    feat = np.array(f['feat'])
    labels = np.array(f['labels'])
    loggin_dir = os.path.join(root_dir, 'runs', 'clustering')
    if not os.path.exists(loggin_dir):
        os.makedirs(loggin_dir)
    logger = EKM_Tensorboard_Logger(loggin_dir)
    data_size = feat.shape[0]
    batch_size = 256
    ekm = EnhancedKMeans(n_clusters=n_clusters, logger=logger, verbose=True, batch_size=batch_size, update_interval=int(data_size / batch_size))
    ekm.fit(feat, labels=labels)

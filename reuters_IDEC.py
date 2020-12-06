import os
import random

import numpy as np
import tensorboard_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.autograd import Variable

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from SDAE import extract_sdae_text, extract_sdae_model
from config import cfg, get_output_dir
from data_loader import Corpus_Loader
from models import ClusterNet
from utils import cluster_acc


class Text_IDEC(object):

    def __init__(self, root_dir, batch_size=256, n_clusters=4, fd_hidden_dim=10, layer_norm=True, lr=0.001,
                 direct_update=False, maxiter=2e4, update_interval=140, tol=0.001, gamma=0.1,
                 fine_tune_infersent=False, use_vat=False, use_tensorboard=False, semi_supervised=False, split_sents=False, id=0, verbose=True, use_ae=True):
        # model's settings
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.fd_hidden_dim = fd_hidden_dim
        self.n_clusters = n_clusters
        self.layer_norm = layer_norm
        self.use_vat = use_vat
        self.semi_supervised = semi_supervised
        self.lr = lr
        self.direct_update = direct_update
        self.maxiter = maxiter
        self.update_interval = update_interval
        self.tol = tol
        self.gamma = gamma
        self.fine_tune_infersent = fine_tune_infersent
        self.verbose = verbose
        self.use_tensorboard = use_tensorboard
        self.id = id
        self.use_cuda = torch.cuda.is_available()
        self.split_sents = split_sents
        self.use_ae = use_ae
        # data loader
        self.corpus_loader = Corpus_Loader(self.root_dir,
                                           layer_norm=self.layer_norm,
                                           verbose=self.verbose,
                                           use_cuda=self.use_cuda,
                                           semi_supervised=self.semi_supervised,
                                           split_sents=self.split_sents)
        # model's components
        self.kmeans = None
        # self.fd_ae = extract_sdae_text(dim=fd_hidden_dim)
        self.fd_ae = extract_sdae_model(input_dim=2000, hidden_dims=cfg.HIDDEN_DIMS)

        self.cluster_layer = None
        self.ae_criteron = nn.MSELoss()
        self.cluster_criteron = F.binary_cross_entropy
        self.optimizer = None
        # model's state
        self.current_p = None
        self.current_q = None
        self.current_pred_labels = None
        self.past_pred_labels = None
        self.current_cluster_acc = None
        # model's logger
        self.logger_tensorboard = None
        # initialize model's parameters and update current state
        self.initialize_model()
        self.initialize_tensorboard()

    def initialize_tensorboard(self):
        outputdir = get_output_dir(self.root_dir)
        loggin_dir = os.path.join(outputdir, 'runs', 'clustering')
        if not os.path.exists(loggin_dir):
            os.makedirs(loggin_dir)
        self.logger_tensorboard = tensorboard_logger.Logger(os.path.join(loggin_dir, '{}'.format(self.id)))

    def initialize_model(self):
        if self.verbose:
            print('Loading pretrainded feedforward autoencoder')
        self.load_pretrained_fd_autoencoder()
        if self.verbose:
            print('Kmeans by hidden features')
        self.initialize_kmeans()
        if self.verbose:
            print('Kmeans cluster acc is {}'.format(self.current_cluster_acc))
            print('Initialzing cluster layer by Kmeans centers')
        self.initialize_cluster_layer()
        if self.verbose:
            print('Initializing Adam optimzer, learning rate is {}'.format(self.lr))
        self.initialize_optimizer()
        if self.verbose:
            print('Updating target distribution')
        self.update_target_distribution()

    def load_pretrained_fd_autoencoder(self):
        """
        load pretrained stack denoise autoencoder
        """
        # outputdir = get_output_dir(self.root_dir)
        outputdir = self.root_dir
        net_filename = os.path.join(outputdir, cfg.PRETRAINED_FAE_FILENAME)
        checkpoint = torch.load(net_filename)
        # there some problems when loading cuda pretrained models
        self.fd_ae.load_state_dict(checkpoint['state_dict'])
        if self.use_cuda:
            self.fd_ae.cuda()

    def initialize_optimizer(self):
        params = [
            {'params': self.fd_ae.parameters()},
            {'params': self.cluster_layer.parameters()}
        ]
        if self.fine_tune_infersent:
            params.append({'params': self.corpus_loader.infersent.parameters(), 'lr': 0.001 * self.lr})
        self.optimizer = optim.Adam(params, lr=self.lr)

    def initialize_kmeans(self):
        features = self.__get_initial_hidden_features()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.current_pred_labels = kmeans.fit_predict(features)
        self.update_cluster_acc()
        self.kmeans = kmeans

    def __get_initial_hidden_features(self):
        batch_size = self.batch_size
        features_numpy = self.corpus_loader.get_fixed_features()
        data_size = self.corpus_loader.data_size
        hidden_feat = np.zeros((data_size, self.fd_hidden_dim))
        for index in range(0, data_size, batch_size):
            data_batch = features_numpy[index: index+batch_size]
            data_batch = Variable(torch.Tensor(data_batch))
            if self.use_cuda:
                data_batch = data_batch.cuda()
            hidden_batch, _ = self.fd_ae(data_batch)
            hidden_batch = hidden_batch.data.cpu().numpy()
            hidden_feat[index: index+batch_size] = hidden_batch
        return hidden_feat

    #################################################################
    def get_current_hidden_features(self):
        return self.__get_initial_hidden_features()
    #################################################################

    def initialize_cluster_layer(self):
        self.cluster_layer = ClusterNet(torch.Tensor(self.kmeans.cluster_centers_.astype(np.float32)))
        if self.use_cuda:
            self.cluster_layer.cuda()

    def get_batch_target_distribution(self, batch_id):
        batch_target_distribution = self.current_p[batch_id]
        batch_target_distribution = Variable(torch.Tensor(batch_target_distribution))
        if self.use_cuda:
            batch_target_distribution = batch_target_distribution.cuda()
        return batch_target_distribution

    def update_target_distribution(self):
        data_size = self.corpus_loader.data_size
        all_q = np.zeros((data_size, self.n_clusters))
        tmp_size = 0
        for current_batch in self.corpus_loader.\
                train_data_iter(self.batch_size):
            id_batch = current_batch[2]
            if self.fine_tune_infersent:
                sent_feat = current_batch[3]
            else:
                sent_feat = current_batch[0]
            hidden_feat, _ = self.fd_ae(sent_feat)
            q_batch = self.cluster_layer(hidden_feat)
            q_batch = q_batch.cpu().data.numpy()
            all_q[id_batch] = q_batch
            tmp_size += len(id_batch)
        assert tmp_size == data_size
        all_p = self.target_distribution_numpy(all_q)
        self.current_p = all_p
        self.current_q = all_q
        self.update_pred_labels()
        self.update_cluster_acc()

    def update_pred_labels(self):
        # warning:
        # When running this function first time,
        # the value of self.past_pred_labels will be equal to self.current_pred_labels
        # This function shouldn't be called for successive times.
        self.past_pred_labels = self.current_pred_labels
        self.current_pred_labels = np.argmax(self.current_q, axis=1)

    def update_cluster_acc(self):
        self.current_cluster_acc = cluster_acc(np.array(self.corpus_loader.train_labels), self.current_pred_labels)

    @staticmethod
    def target_distribution_torch(q):
        p = torch.pow(q, 2) / torch.sum(q, dim=0).unsqueeze(0)
        p = p / torch.sum(p, dim=1).unsqueeze(1)
        # p = torch.t(torch.t(p) / torch.sum(p, dim=1))
        return Variable(p.data)

    @staticmethod
    def target_distribution_numpy(q):
        p = np.power(q, 2) / np.sum(q, axis=0, keepdims=True)
        p = p / np.sum(p, axis=1, keepdims=True)
        return p

    def vat(self, x_batch, xi=0.1, Ip=1):
        # virtual adversarial training
        # forbid x_batch's grad backward
        x_batch = Variable(x_batch.data)
        hidden_batch, _ = self.fd_ae(x_batch)
        q_batch = self.cluster_layer(hidden_batch)
        q_batch = Variable(q_batch.data)
        # initialize residue d to normalized random vector
        d = torch.randn(x_batch.size())
        if self.use_cuda:
            d = d.cuda()
        d = d / (torch.norm(d, p=2, dim=1, keepdim=True) + 1e-8)
        # ensure model's parameter to be 0
        self.model_zero_grad()
        for i in range(Ip):
            d = nn.Parameter(d)
            tmp_x_batch = x_batch + xi * d
            tmp_hidden_batch, _ = self.fd_ae(tmp_x_batch)
            tmp_q_batch = self.cluster_layer(tmp_hidden_batch)
            tmp_loss = F.binary_cross_entropy(tmp_q_batch, q_batch)
            tmp_loss.backward()
            d = d.grad.data
            d = d / (torch.norm(d, p=2, dim=1, keepdim=True) + 1e-8)
            self.model_zero_grad()
        # computing vat loss
        d = Variable(d)
        tmp_x_batch = x_batch + xi * d
        tmp_hidden_batch, _ = self.fd_ae(tmp_x_batch)
        tmp_q_batch = self.cluster_layer(tmp_hidden_batch)
        tmp_loss = F.binary_cross_entropy(tmp_q_batch, q_batch)
        return tmp_loss

    def whether_convergence(self):
        delta_label = np.sum(self.past_pred_labels != self.current_pred_labels) / float(len(self.current_pred_labels))
        return delta_label < self.tol

    def model_zero_grad(self):
        self.cluster_layer.zero_grad()
        self.fd_ae.zero_grad()
        if self.fine_tune_infersent:
            self.corpus_loader.infersent.zero_grad()

    def clustering(self):
        if self.semi_supervised:
            train_data_iter = self.corpus_loader.train_data_iter(self.batch_size,
                                                                 return_variable_features=self.fine_tune_infersent,
                                                                 shuffle=False,
                                                                 infinite=True)
            constraints_data_iter = self.corpus_loader.constraint_data_iter(self.batch_size,
                                                                 shuffle=True,
                                                                 infinite=True)
            ite = 0
            tmp_ite_cons = 0
            while True:
                if random.random() > 0.95:
                    self.model_zero_grad()
                    feat_batch1, feat_batch2 = constraints_data_iter.next()
                    hidden_batch1, output_feat1 = self.fd_ae(feat_batch1)
                    hidden_batch2, output_feat2 = self.fd_ae(feat_batch2)
                    ae_loss1 = self.ae_criteron(output_feat1, feat_batch1)
                    ae_loss2 = self.ae_criteron(output_feat2, feat_batch2)
                    q_batch1 = self.cluster_layer(hidden_batch1)
                    q_batch2 = self.cluster_layer(hidden_batch2)
                    if random.random() > 0.5:
                        q_batch1, q_batch2 = q_batch2, q_batch1
                    q_batch2 = Variable(q_batch2.data)
                    k_loss = self.cluster_criteron(q_batch1, q_batch2)
                    loss = 2 * self.gamma * k_loss + ae_loss1 + ae_loss2

                    if self.use_tensorboard:
                        self.logger_tensorboard.log_value('cons_loss', loss.data[0], tmp_ite_cons)
                        self.logger_tensorboard.log_value('cons_kl_loss', k_loss.data[0], tmp_ite_cons)
                    loss.backward()
                    self.optimizer.step()
                    tmp_ite_cons += 1
                else:
                    if ite % self.update_interval == (self.update_interval - 1):
                        self.update_target_distribution()
                        print('Iter {} acc {}'.format(ite, self.current_cluster_acc))
                        if self.use_tensorboard:
                            self.logger_tensorboard.log_value('acc', self.current_cluster_acc, ite)
                        if ite > 0 and self.whether_convergence():
                            break

                    # current_batch = train_data_iter.next()
                    current_batch = next(train_data_iter)
                    fixed_feat_batch = current_batch[0]
                    id_batch = current_batch[2]
                    if self.fine_tune_infersent:
                        sent_feat_batch = current_batch[3]
                    else:
                        sent_feat_batch = fixed_feat_batch

                    self.model_zero_grad()
                    hidden_batch, output_batch = self.fd_ae(sent_feat_batch)
                    q_batch = self.cluster_layer(hidden_batch)
                    if self.direct_update:
                        p_batch = self.target_distribution_torch(q_batch)
                    else:
                        p_batch = self.get_batch_target_distribution(id_batch)
                    ae_loss = self.ae_criteron(output_batch, fixed_feat_batch)
                    cluster_loss = self.cluster_criteron(q_batch, p_batch)
                    if self.use_vat:
                        vat_loss = self.vat(sent_feat_batch)
                    else:
                        vat_loss = 0
                    loss = self.gamma * (cluster_loss + vat_loss) + ae_loss
                    if self.use_tensorboard:
                        self.logger_tensorboard.log_value('cluster_loss', cluster_loss.data[0], ite)
                        self.logger_tensorboard.log_value('ae_loss', ae_loss.data[0], ite)
                        if self.use_vat:
                            self.logger_tensorboard.log_value('vat_loss', vat_loss.data[0], ite)
                        self.logger_tensorboard.log_value('loss', loss.data[0], ite)
                    loss.backward()
                    self.optimizer.step()
                    ######################################
                    ite += 1
                    if ite >= int(self.maxiter):
                        break
                    ######################################
        else:
            train_data_iter = self.corpus_loader.train_data_iter(self.batch_size,
                                                                 # return_variable_features=self.fine_tune_infersent,
                                                                 shuffle=False,
                                                                 infinite=True)
            for ite in range(int(self.maxiter)):
                if ite % self.update_interval == (self.update_interval - 1):
                    self.update_target_distribution()
                    print('Iter {} acc {}'.format(ite, self.current_cluster_acc))
                    if self.use_tensorboard:
                        self.logger_tensorboard.log_value('acc', self.current_cluster_acc, ite)
                    if ite > 0 and self.whether_convergence():
                        break

                # current_batch = train_data_iter.next()
                current_batch = next(train_data_iter)
                fixed_feat_batch = current_batch[0]
                id_batch = current_batch[2]
                if self.fine_tune_infersent:
                    sent_feat_batch = current_batch[3]
                else:
                    sent_feat_batch = fixed_feat_batch

                self.model_zero_grad()
                hidden_batch, output_batch = self.fd_ae(sent_feat_batch)
                q_batch = self.cluster_layer(hidden_batch)
                if self.direct_update:
                    p_batch = self.target_distribution_torch(q_batch)
                else:
                    p_batch = self.get_batch_target_distribution(id_batch)
                if self.use_ae:
                    ae_loss = self.ae_criteron(output_batch, fixed_feat_batch)
                else:
                    ae_loss = 0
                cluster_loss = self.cluster_criteron(q_batch, p_batch)
                if self.use_vat:
                    vat_loss = self.vat(sent_feat_batch)
                else:
                    vat_loss = 0
                loss = self.gamma * (cluster_loss + vat_loss) + ae_loss
                if self.use_tensorboard:
                    self.logger_tensorboard.log_value('cluster_loss', cluster_loss.data[0], ite)
                    if self.use_ae:
                        self.logger_tensorboard.log_value('ae_loss', ae_loss.data[0], ite)
                    if self.use_vat:
                        self.logger_tensorboard.log_value('vat_loss', vat_loss.data[0], ite)
                    self.logger_tensorboard.log_value('loss', loss.data[0], ite)
                loss.backward()
                self.optimizer.step()

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
    with open('idec_results.txt','a') as f:
        import json
        f.write(json.dumps(tmp))
        f.write('\n')
if __name__ == '__main__':
    data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers', 3:'reuters_2', 4:'reuters_5', 5:'reuters_10', 6:'reuters_19'}
    n_cluster_dict = {0: 4, 1: 14, 2: 10, 3:2, 4:5, 5:10, 6:19}
    raw_feat_name = 'DEC'
    trial_num = 10
    
    for corpora_id in range(3, 7):
        corpora_name = data_dict[corpora_id]
        root_dir = 'data/' + data_dict[corpora_id]
        n_clusters = n_cluster_dict[corpora_id]
        for use_ae in [True, False]:
            feat_name = 'I'+raw_feat_name if use_ae else raw_feat_name
            best_acc = 0.0
            best_pred = None
            all_pred = []
            all_acc = []
            all_nmi = []
            all_ari = []
            for i in range(trial_num):

                text_idec_model = Text_IDEC(root_dir=root_dir + '/tfidf_i',
                                            update_interval=10,
                                            n_clusters=n_clusters,
                                            use_tensorboard=True,
                                            use_vat=False,
                                            id=4,
                                            semi_supervised=False,
                                            split_sents=True,
                                            use_ae=use_ae,
                                            fd_hidden_dim=cfg.HIDDEN_DIMS[-1])
                text_idec_model.clustering()
                print('Total acc is {}'.format(text_idec_model.current_cluster_acc))
                pred = np.array(text_idec_model.current_pred_labels)
                labels = np.array(text_idec_model.corpus_loader.train_labels)
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
            print('{} best acc is {}'.format(feat_name, best_acc))
            pred_std = np.std(all_acc)
            pred_mean = np.mean(all_acc)
            dump_mongo(corpora=corpora_name, 
                    feat_name=feat_name, 
                    n_topics=n_clusters, 
                    acc=best_acc, 
                    pred=best_pred.tolist(), 
                    all_pred=all_pred, 
                    all_acc=all_acc, 
                    all_nmi=all_nmi, 
                    all_ari=all_ari)

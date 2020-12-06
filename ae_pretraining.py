from __future__ import print_function

import argparse
import os

# used for logging to TensorBoard
import tensorboard_logger as TF_LOGGER
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.backends import cudnn as cudnn
from torch.utils import data as data
import random
import numpy as np

from SDAE import sdae_model
from config import cfg


def initialize_environment(random_seed=50, use_cuda=torch.cuda.is_available()):
    # Set the seed for reproducing the results
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

class EncodedTextDataset(data.Dataset):
    """Custom dataset loader for Pretraining SDAE"""
    def __init__(self, root, feat_name='', feat_func=lambda x:x, train=True, verbose=True):
        self.root_dir = root
        self.train = train
        self.verbose = verbose

        if self.train:
            if feat_name == '':
                train_feat_path = os.path.join(self.root_dir, cfg.TRAIN_TEXT_FEAT_FILE_NAME)
            else:
                train_feat_path = os.path.join(self.root_dir, feat_name+'.h5')
            self.train_data, self.train_labels, _ = load_feat(train_feat_path)
            self.train_data = feat_func(self.train_data)
            self.train_ids = np.array(range(len(self.train_labels)))
            if self.verbose:
                print('Loading {} training item'.format(len(self.train_labels)))
        else:
            if feat_name == '':
                test_feat_path = os.path.join(self.root_dir, cfg.TEST_TEXT_FEAT_FILE_NAME)
            else:
                test_feat_path = os.path.join(self.root_dir, feat_name+'.h5')
            self.test_data, self.test_labels, _ = load_feat(test_feat_path)
            self.test_data = feat_func(self.test_data)
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

def ln(feat):
    return (feat - feat.mean(axis=1, keepdims=True)) / feat.std(axis=1, keepdims=True)

def norm(feat):
    return feat / np.linalg.norm(feat, axis=1, keepdims=True)

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch SDAE Training')
parser.add_argument('--corpora_id', type=int, default=0, help='the id of corpora')
parser.add_argument('--feat_id', type=int, default=0, help='the id of feat')
parser.add_argument('--batchsize', type=int, default=256, help='batch size used for pretraining')
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs used for pretraining')
parser.add_argument('--step_epoch', type=int, default=80,
                    help='stepsize in terms of number of epoch for pretraining. lr is decreased by 10 after every stepsize.')

# Note: The learning rate of pretraining stage differs for each dataset.
# As noted in the paper, it depends on the original dimension of the data samples.
# This is purely selected such that the SDAE's are trained with maximum possible learning rate for each dataset.
# We set mnist,reuters,rcv1=10, ytf=1, coil100,yaleb=0.1
# For convolutional SDAE lr if fixed to 0.1
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate for pretraining')
parser.add_argument('--dropout', type=float, help='dropout of SDAE', default=0.2)
parser.add_argument('--id', type=int, help='identifying number for storing tensorboard logs')


data_dict = {0:'ag_news',1:'dbpedia', 2:'yahoo_answers'}
feat_dict = {0:'infersent',1:'elmo_max', 2:'elmo_mean', 3:'tfidf'}
input_feat_size_dict = {0: 4096,1:1024,2:1024, 3:2000}
feat_func_dict = {'ln': ln, 'n': norm, 'i': lambda x: x}

tensorboard_logger = 0
def main():
    global args
    global tensorboard_logger
    use_cuda = torch.cuda.is_available()
    initialize_environment(random_seed=cfg.RNG_SEED, use_cuda=use_cuda)

    args = parser.parse_args()
    assert 0 <= args.feat_id <= 3
    assert 0 <= args.corpora_id <= 2
    feat_name = feat_dict[args.feat_id]
    corpora_name = data_dict[args.corpora_id]
    datadir = os.path.join('data', corpora_name)
    nepoch = args.nepoch
    step = args.step_epoch
    dropout = args.dropout
    n_layers = cfg.N_LAYERS
    input_dim = input_feat_size_dict[args.feat_id]
    hidden_dims = cfg.HIDDEN_DIMS
    if args.feat_id == 0 or args.feat_id == 1:
        feat_func_dict = {'ln': ln, 'n': norm}
    elif args.feat_id == 2:
        feat_func_dict = {'ln': ln, 'n': norm, 'i': lambda x: x}
    elif args.feat_id == 3:
        feat_func_dict = {'i': lambda x: x}


    for feat_func_name, feat_func in feat_func_dict.items():
        print( corpora_name,feat_name, feat_func_name)
        outputdir = os.path.join('data', corpora_name, feat_name+'_'+feat_func_name)
        # logging information
        loggin_dir = os.path.join(outputdir, 'runs', 'pretraining')
        if not os.path.exists(loggin_dir):
            os.makedirs(loggin_dir)
        tensorboard_logger = TF_LOGGER.Logger(os.path.join(loggin_dir, '%s' % (args.id)))
        # tensorboard_logger.configure(os.path.join(loggin_dir, '%s' % (args.id)))

        trainset = EncodedTextDataset(root=datadir, train=True, feat_name=feat_name, feat_func=feat_func)
        testset = EncodedTextDataset(root=datadir, train=False, feat_name=feat_name, feat_func=feat_func)
        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, **kwargs)

        pretrain(outputdir,
                 {'nlayers':n_layers,
                  'dropout':dropout,
                  'reluslope':0.0,
                  'nepoch':nepoch,
                  'lrate':[args.lr],
                  'wdecay':[0.0],
                  'step':step,
                  'input_dim':input_dim,
                  'hidden_dims':hidden_dims},
                 use_cuda,
                 trainloader,
                 testloader)


def pretrain(outputdir, params, use_cuda, trainloader, testloader):
    numlayers = params['nlayers']
    lr = params['lrate'][0]
    maxepoch = params['nepoch']
    stepsize = params['step']
    input_dim = params['input_dim']
    hidden_dims = params['hidden_dims']
    startlayer = 0

    # For simplicity, I have created placeholder for each datasets and model
    # net = sdae_text(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    net = sdae_model(input_dim=input_dim, hidden_dims=hidden_dims, dropout=params['dropout'], slope=params['reluslope'])

    # For the final FT stage of SDAE pretraining, the total epoch is twice that of previous stages.
    maxepoch = [maxepoch]*numlayers + [maxepoch*2]
    stepsize = [stepsize]*(numlayers+1)

    if use_cuda:
        net.cuda()

    for index in range(startlayer, numlayers+1):
        # Freezing previous layer weights
        if index < numlayers:
            for par in net.base[index].parameters():
                par.requires_grad = False
        else:
            for par in net.base[numlayers-1].parameters():
                par.requires_grad = True

        # setting up optimizer - the bias params should have twice the learning rate w.r.t. weights params
        bias_params = filter(lambda x: ('bias' in x[0]) and (x[1].requires_grad), net.named_parameters())
        bias_params = list(map(lambda x: x[1], bias_params))
        nonbias_params = filter(lambda x: ('bias' not in x[0]) and (x[1].requires_grad), net.named_parameters())
        nonbias_params = list(map(lambda x: x[1], nonbias_params))

        optimizer = optim.SGD([{'params': bias_params, 'lr': 2*lr}, {'params': nonbias_params}],
                              lr=lr, momentum=0.9, weight_decay=params['wdecay'][0], nesterov=True)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize[index], gamma=0.1)

        print('\nIndex: %d \t Maxepoch: %d'%(index, maxepoch[index]))

        for epoch in range(maxepoch[index]):
            scheduler.step()
            train(trainloader, net, index, optimizer, epoch, use_cuda)
            test(testloader, net, index, epoch, use_cuda)
            # Save checkpoint
            save_checkpoint({'epoch': epoch+1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                            index, filename=outputdir, n_layers=numlayers)


# Training
def train(trainloader, net, index, optimizer, epoch, use_cuda):
    losses = AverageMeter()

    print('\nIndex: %d \t Epoch: %d' %(index,epoch))

    net.train()

    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs_Var = Variable(inputs)
        outputs = net(inputs_Var, index)

        # record loss
        losses.update(outputs.data[0], inputs.size(0))

        outputs.backward()
        optimizer.step()

    # log to TensorBoard
    tensorboard_logger.log_value('train_loss_{}'.format(index), losses.avg, epoch)


# Testing
def test(testloader, net, index, epoch, use_cuda):
    losses = AverageMeter()

    net.eval()

    for batch_idx, (inputs, targets, _) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs_Var = Variable(inputs, volatile=True)
        outputs = net(inputs_Var, index)

        # measure accuracy and record loss
        losses.update(outputs.data[0], inputs.size(0))

    # log to TensorBoard
    tensorboard_logger.log_value('val_loss_{}'.format(index), losses.avg, epoch)


# Saving checkpoint
def save_checkpoint(state, index, filename, n_layers):
    if index >= n_layers:
        torch.save(state, os.path.join(filename, cfg.PRETRAINED_FAE_FILENAME))
    else:
        torch.save(state, filename+'/checkpoint_%d.pth.tar' % index)

if __name__ == '__main__':
    main()

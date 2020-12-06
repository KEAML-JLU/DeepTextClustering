from __future__ import print_function

import argparse
import os

# used for logging to TensorBoard
import tensorboard_logger
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils import data as data

from SDAE import sdae_model
from config import cfg, get_output_dir, AverageMeter
# Parse all the input argument
from data_loader import EncodedTextDataset
from utils import initialize_environment

parser = argparse.ArgumentParser(description='PyTorch SDAE Training')
parser.add_argument('--batchsize', type=int, default=256, help='batch size used for pretraining')
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs used for pretraining')
parser.add_argument('--step_epoch', type=int, default=80,
                    help='stepsize in terms of number of epoch for pretraining. lr is decreased by 10 after every stepsize.')

# Note: The learning rate of pretraining stage differs for each dataset.
# As noted in the paper, it depends on the original dimension of the data samples.
# This is purely selected such that the SDAE's are trained with maximum possible learning rate for each dataset.
# We set mnist,reuters,rcv1=10, ytf=1, coil100,yaleb=0.1
# For convolutional SDAE lr if fixed to 0.1
parser.add_argument('--lr', default=1, type=float, help='initial learning rate for pretraining')
parser.add_argument('--data_dir', dest='db_dir', type=str, default='data/ag_news', help='directory of dataset')
parser.add_argument('--dropout', type=float, help='dropout of SDAE', default=0.2)
parser.add_argument('--id', type=int, help='identifying number for storing tensorboard logs')


def main():
    global args
    use_cuda = torch.cuda.is_available()
    initialize_environment(random_seed=cfg.RNG_SEED, use_cuda=use_cuda)

    args = parser.parse_args()
    datadir = args.db_dir
    outputdir = get_output_dir(args.db_dir)
    nepoch = args.nepoch
    step = args.step_epoch
    dropout = args.dropout
    n_layers = cfg.N_LAYERS
    input_dim = cfg.INPUT_DIM
    hidden_dims = cfg.HIDDEN_DIMS

    # logging information
    loggin_dir = os.path.join(outputdir, 'runs', 'pretraining')
    if not os.path.exists(loggin_dir):
        os.makedirs(loggin_dir)
    tensorboard_logger.configure(os.path.join(loggin_dir, '%s' % (args.id)))

    trainset = EncodedTextDataset(root=datadir, train=True)
    testset = EncodedTextDataset(root=datadir, train=False)
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

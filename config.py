import os.path as osp
from easydict import EasyDict as edict

__C = edict()


#######
# OPTIONS FROM RCC CODE
#######
__C.RCC = edict()

__C.RCC.NOISE_THRESHOLD = 0.01

__C.RCC.MAX_NUM_SAMPLES_DELTA = 250

__C.RCC.MIN_RATIO_SAMPLES_DELTA = 0.01

__C.RCC.GNC_DATA_START_POINT = 132*16

#######
# MISC OPTIONS
#######


# Root directory
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__),'..'))

# size of the dataset
__C.SAMPLES = 70000


# Number of pairs per batch
__C.PAIRS_PER_BATCH = 128


class Config(object):
    def __init__(self):
        # For reproducibility
        self.RNG_SEED = 50

        self.TRAIN_DATA_NAME = 'train'
        self.TEST_DATA_NAME = 'test'
        self.CONSTRAINTS_NAME = 'constraints'
        self.GLOVE_PATH = 'data/glove.840B.300d.txt'
        self.INFERSENT_PATH = 'data/infersent.allnli.pickle'
        self.PRETRAINED_FAE_FILENAME = 'ae_checkpoint.pth.tar'
        #############################
        # modify in 2019 4 18
        # self.TRAIN_TEXT_FEAT_FILE_NAME= 'train_text_feat.h5'
        self.TRAIN_TEXT_FEAT_FILE_NAME= '../tfidf.h5'
        #############################
        self.TEST_TEXT_FEAT_FILE_NAME= 'test_text_feat.h5'
        self.SEED_FILE_NAME= 'seed.txt'

        # embedding dimension
        self.DIM = 10
        #############################
        # modify in 2019 4 18
        # self.INPUT_DIM = 4096
        self.INPUT_DIM = 2000 
        #############################
        #############################
        # modify in 2019 4 18
        # self.HIDDEN_DIMS = [1200, 1200]
        self.HIDDEN_DIMS = [1200, 1200]
        #############################
        self.N_LAYERS = len(self.HIDDEN_DIMS)

        # Fraction of "change in label assignment of pairs" to be considered for stopping criterion - 1% of pairs
        self.STOPPING_CRITERION = 0.001

cfg = Config()



def get_data_dir(db):
    """
    :param db:
    :return: path to data directory
    """
    # path = osp.abspath(osp.join(__C.ROOT_DIR, 'data', db))
    path = db
    return path

def get_output_dir(db):
    """
    :param db:
    :return: path to data directory
    """
    # path = osp.abspath(osp.join(__C.ROOT_DIR, 'data', db, 'results'))
    path = osp.join(db, 'results')
    return path

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


import logging
import os
import sys
import numpy as np
from pytz import timezone
from datetime import datetime
import itertools
import torch
import shutil

import torch.nn as nn
import torch.nn.functional as F
import random

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_data_loader(data_path="", data_path2="", data_type="live", batch_size=5, shuffle=True, drop_last=True):
    data = None
    live_spoof_label = None
    
    if data_type == "live":
        data = np.load(data_path)
        live_spoof_label = np.ones(len(data), dtype=np.int64)

    else:
        print_data = np.load(data_path)
        replay_data = np.load(data_path2)
        data = np.concatenate((print_data, replay_data), axis=0)
        live_spoof_label = np.zeros(len(data), dtype=np.int64)

    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label))

    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader


def get_inf_iterator(data_loader):
    while True:
        for images, live_spoof_labels in data_loader:
            yield (images, live_spoof_labels)


def logger(root_dir, results_filename, train=True):
    logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
    mkdir(root_dir)
    if train:
        file_handler = logging.FileHandler(filename= root_dir + results_filename +'_train.log')
    else:
        file_handler = logging.FileHandler(filename= root_dir + results_filename +'_test.log')

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    date = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)


def Find_Optimal_Cutoff(TPR, FPR, threshold): 
    y = TPR + (1 - FPR)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def get_testset(data_path="", data_path2="", data_type="live"):
    data = None
    live_spoof_label = None
    
    if data_type == "live":
        data = np.load(data_path)
        live_spoof_label = np.ones(len(data), dtype=np.int64)
    elif data_type == "PAspoof":
        print_data = np.load(data_path)
        replay_data = np.load(data_path2)
        data = np.concatenate((print_data, replay_data), axis=0)
        live_spoof_label = np.zeros(len(data), dtype=np.int64)
    else:
        data = np.load(data_path)
        live_spoof_label = np.zeros(len(data), dtype=np.int64)

    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label))
    return trainset

def get_3Ddata_loader(data_path="", data_type="live", batch_size=5, shuffle=True, drop_last=True):
    data = None
    live_spoof_label = None
    
    if data_type == "live":
        data = np.load(data_path)
        live_spoof_label = np.ones(len(data), dtype=np.int64)

    else:
        data = np.load(data_path)
        live_spoof_label = np.zeros(len(data), dtype=np.int64)

    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label))
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader

def get_nearest_neighbor(X, Y, n, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
        n: the numbers of nearest neighbor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")
    _, idx = torch.sort(distances,dim=1)
    nei_idx = idx[:,:n]
    # neightbor = distances[:,:2]
    return nei_idx

def seed_init(args):
    if args.protocol=='seen_attack':
        if args.trained_model=='OCR':
            seed = 1003510260
        elif args.trained_model=='OCM':
            seed = 122
        elif args.trained_model=='CMR':
            seed = 1008
        else:
            seed = 748235357
    elif args.protocol=='unseen_attack':
        if args.val_dataset=='3DMAD':
            if args.trained_model=='OCR':
                seed = 110
            elif args.trained_model=='OCM':
                seed = 38
            elif args.trained_model=='CMR':
                seed = 333532615
            else:
                seed = 5031
        if args.val_dataset=='HKBUv1+':
            if args.trained_model=='OCR':
                seed = 1753
            elif args.trained_model=='OCM':
                seed = 10
            elif args.trained_model=='CMR':
                seed = 1443
            else:
                seed = 19
    else:
        if args.trained_model=='OMR_replay':
            seed = 453318365
        elif args.trained_model=='OMR_print':
            seed = 657974677
        else:
            seed = 492041455

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    return seed




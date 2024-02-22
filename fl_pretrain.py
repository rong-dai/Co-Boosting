#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import sys
sys.path.append('/gdata/dairong/DENSE-main/pylib')
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import pdb
import logging
import registry
from utils_fl import *
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

max_norm = 10

def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # console = logging.StreamHandler()
    # console.setLevel(logging.DEBUG)
    # console.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(console)
    return logger

warnings.filterwarnings('ignore')

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        self.logger = logger

    def update_weights(self, model, client_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.local_lr, momentum=0.9)
        local_acc_list = []
        for iter in range(self.args.local_ep):
            acc = 0; train_loss = 0; total_num = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                # ---------------------------------------
                output = model(images)
                loss = F.cross_entropy(output, labels)
                acc += torch.sum(output.max(dim=1)[1] == labels).item()
                total_num += len(labels)
                train_loss += loss.item()
                # ---------------------------------------
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
                optimizer.step()
            train_loss = train_loss / total_num; acc = acc / total_num
            logger.info('Iter:{:4d} Train_set: Average loss: {:.4f}, Accuracy: {:.4f}'
                    .format(iter, train_loss, acc))
            local_acc_list.append(acc)
        return model.state_dict(), np.array(local_acc_list)

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--local_lr', type=float, default=0.01,
                        help='learning rate')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    parser.add_argument('--sigma', default=0.0, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    # Default
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='models for each client.')
    parser.add_argument('--identity', default="", type=str,
                        help='identity.')
    parser.add_argument('--logidentity', default="", type=str,
                        help='logidentity.')
    parser.add_argument('--imgsize', default=32, type=int,
                        help='img_size')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batchsize')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = args_parser()
    args.identity = args.dataset + "_clients" + str(args.num_users) + "_" + str(args.partition) + str(args.beta) + "_sig" + str(args.sigma)
    args.identity += "_" + args.model + "_Llr" + str(args.local_lr) + "_Le" + str(args.local_ep) + "_seed" + str(args.seed)
    args.logidentity = args.identity

    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    logpath_prefix = '/gdata/dairong/Co_boosting/LOG/FL_pretrain/'
    if not os.path.exists(logpath_prefix):
        os.makedirs(logpath_prefix)
    _log_path = os.path.join(cur_dir, logpath_prefix +  args.logidentity+'.log')
    _logging_name = args.logidentity
    logger = logger_config(log_path=_log_path, logging_name=_logging_name)
    logger.info(args)

    ############################################
    # Setup dataset
    ############################################
    setup_seed(args.seed)
    num_classes, train_dataset, test_dataset = registry.get_dataset(name=args.dataset, data_root='/gdata/dairong/fedsam/Data/Raw')
    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        train_dataset,test_dataset, args.partition, beta=args.beta, num_users=args.num_users, logger=logger, args=args)

    _sum = 0
    for i in range(len(traindata_cls_counts)):
        _cnt = 0
        for key in traindata_cls_counts[i].keys():
            _cnt += traindata_cls_counts[i][key]
        logger.info(_cnt)
        _sum += _cnt
    logger.info(_sum)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                shuffle=False, num_workers=4)
    # Build models
    global_model = registry.get_model(args.model, num_classes=num_classes)
    global_model = global_model.cuda()

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    acc_list = []
    users = []
    for idx in range(args.num_users):
        logger.info("client {}".format(idx))
        users.append("client_{}".format(idx))
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx],logger=logger)
        w, local_acc = local_model.update_weights(copy.deepcopy(global_model), idx)
        acc_list.append(local_acc)
        local_weights.append(copy.deepcopy(w))

    ## save models
    save_path_prefix = '/gdata/dairong/Co_boosting/checkpoints/FL_pretrain/'
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    torch.save(local_weights, save_path_prefix + '{}.pkl'.format(args.identity))

    ## test FedAvg model
    global_model = registry.get_model(args.model, num_classes=num_classes)
    global_model = global_model.cuda()
    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)
    test_acc, test_loss = class_test(global_model, test_loader, logger)
    for idx in range(args.num_users):
        logger.info('Test acc of Client ID {:3d}, {:.4f}'.format(idx, acc_list[idx][-1]))
    logger.info('FedAvg global model acc: Average loss: {:.4f}, Accuracy: {:.4f}'
          .format(test_loss, test_acc))

    ## test Direct Ensemble model
    model_list = []
    for i in range(len(local_weights)):
        net = copy.deepcopy(global_model)
        net.load_state_dict(local_weights[i])
        model_list.append(net)
    ensemble_model = Ensemble(model_list)
    test_acc, test_loss = class_test(ensemble_model, test_loader, logger)
    logger.info('Direct Ensemble model acc: Average loss: {:.4f}, Accuracy: {:.4f}'
          .format(test_loss, test_acc))
        # ===============================================
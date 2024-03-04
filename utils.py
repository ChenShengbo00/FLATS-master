import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from itertools import product
import math
import copy
import time
import logging
import pickle
import random
import sys

from datasets import TS_truncated
#, MNIST_truncated, EMNIST_truncated, CIFAR10_truncated, CIFAR10_Poisoned, CIFAR10NormalCase_truncated, EMNIST_NormalCase_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # 可以选择"w"
        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_emnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    emnist_train_ds = EMNIST_truncated(datadir, train=True, download=True, transform=transform)
    emnist_test_ds = EMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = emnist_train_ds.data, emnist_train_ds.target
    X_test, y_test = emnist_test_ds.data, emnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])
    # 读取数据
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_UCR_data(datadir, dataset):
    train_data = np.loadtxt(datadir + dataset +'/'+dataset + '_TRAIN.txt')
    test_data = np.loadtxt(datadir + dataset + '/' + dataset + '_TEST.txt')

    X_train, y_train = train_data[:, 1:],train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
    print('load UCR data', X_train.shape, X_test.shape)
    return (X_train, y_train, X_test, y_test)

def partition_tsdata(dataset, datadir, partition, n_nets, alpha):
    # partition_strategy = "homo"
    # partition_strategy = "hetero-dir"
    print('---------------load UCR daset-------------------')
    X_train, y_train, X_test, y_test = load_UCR_data(datadir, dataset)
    n_train = X_train.shape[0]
    if partition == "homo":
        idxs = np.random.permutation(n_train) # 随机排序
        batch_idxs = np.array_split(idxs, n_nets) # 把idxs分为 nnets份
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)} #用户id->数据的字典
    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        K = len(np.unique(y_train))
        N = y_train.shape[0]
        net_dataidx_map = {}
        while (min_size < 1) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0] # 取出10个类分别对应的下标集合
                np.random.shuffle(idx_k) # 打乱下标，重复alphanets次
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets)) # 地雷克雷分布
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum() # 归一化
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # 数据采用地雷克雷分布分配给用户
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    # fanhui
    return net_dataidx_map
def partition_data(dataset, datadir, partition, n_nets, alpha, args):
    # partition_strategy = "homo"
    # partition_strategy = "hetero-dir"
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'emnist':
        X_train, y_train, X_test, y_test = load_emnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset.lower() == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        # if args.poison_type == "howto":
        #     sampled_indices_train = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
        #                                 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389]
        #     sampled_indices_test = [32941, 36005, 40138]
        #     cifar10_whole_range = np.arange(X_train.shape[0])
        #     remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_train+sampled_indices_test]
        #     X_train = X_train[sampled_indices_train, :, :, :]
        #     logger.info("@@@ Poisoning type: {} Num of Remaining Data Points (excluding poisoned data points): {}".format(
        #                                 args.poison_type, 
        #                                 X_train.shape[0]))
        
        # # 0-49999 normal cifar10, 50000 - 50735 wow airline
        # if args.poison_type == 'southwest+wow':
        #     with open('./saved_datasets/wow_images_new_whole.pkl', 'rb') as train_f:
        #         saved_wow_dataset_whole = pickle.load(train_f)
        #     X_train = np.append(X_train, saved_wow_dataset_whole, axis=0)
        n_train = X_train.shape[0]

    elif dataset == 'cinic10':
        _train_dir = './data/cinic10/cinic-10-trainlarge/train'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), 
                                                                                            requires_grad=False),
                                                                                            (4,4,4,4),mode='reflect').data.squeeze()),
                                                            transforms.ToPILImage(),
                                                            transforms.RandomCrop(32),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=cinic_mean,std=cinic_std),
                                                            ]))
        y_train = trainset.get_train_labels
        n_train = y_train.shape[0]
    elif dataset == "shakespeare":
        net_dataidx_map = {}
        with open(datadir[0]) as json_file:
            train_data = json.load(json_file)

        with open(datadir[1]) as json_file:
            test_data = json.load(json_file)

        for j in range(n_nets):
            client_user_name = train_data["users"][j]

            client_train_data = train_data["user_data"][client_user_name]['x']
            num_samples_train = len(client_train_data)
            net_dataidx_map[j] = [i for i in range(num_samples_train)] # TODO: this is a dirty hack. needs modification
        return None, net_dataidx_map, None
    else:
        print('---------------load UCR daset-------------------')
        X_train, y_train, X_test, y_test = load_UCR_data(datadir, dataset)
        n_train = X_train.shape[0]
    # 为用户分配数据
    if partition == "homo":
        idxs = np.random.permutation(n_train) # 随机排序
        batch_idxs = np.array_split(idxs, n_nets) # 把idxs分为 nnets份
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)} #用户id->数据的字典

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        K = len(np.unique(y_train))
        N = y_train.shape[0]
        net_dataidx_map = {}
        while (min_size < K) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0] # 取出10个类分别对应的下标集合
                np.random.shuffle(idx_k) # 打乱下标，重复alphanets次
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets)) # 地雷克雷分布
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum() # 归一化
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # 数据采用地雷克雷分布分配给用户
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        if dataset == 'cifar10':
            if args.poison_type == 'howto' or args.poison_type == 'greencar-neo':
                green_car_indices = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365, 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
                #sanity_check_counter = 0
                for k, v in net_dataidx_map.items():
                    remaining_indices = [i for i in v if i not in green_car_indices]
                    #sanity_check_counter += len(remaining_indices)
                    net_dataidx_map[k] = remaining_indices

            #logger.info("Remaining total number of data points : {}".format(sanity_check_counter))
            # sanity check:
            #aggregated_val = []
            #for val in net_dataidx_map.values():
            #    aggregated_val+= val
            #black_box_indices = [i for i in range(50000) if i not in aggregated_val]
            #logger.info("$$$$$$$$$$$$$$ recovered black box indices: {}".format(black_box_indices))
            #exit()
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return net_dataidx_map

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ('mnist', 'emnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        if dataset == 'emnist':
            dl_obj = EMNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl

def get_ts_loader(dataset, datadir, train_bs, test_bs, dataidxs=None):

    dl_obj = TS_truncated

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True)
    test_ds = dl_obj(datadir, train=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    # print('get ts loader length: ', len(train_ds), len(test_ds))

    return train_dl, test_dl

def get_dataloader_normal_case(dataset, datadir, train_bs, test_bs, 
                                dataidxs=None, 
                                user_id=0, 
                                num_total_users=200,
                                poison_type="southwest",
                                ardis_dataset=None,
                                attack_case='normal-case'):
    if dataset in ('mnist', 'emnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        if dataset == 'emnist':
            dl_obj = EMNIST_NormalCase_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        elif dataset == 'cifar10':
            dl_obj = CIFAR10NormalCase_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])

        # this only supports cifar10 right now, please be super careful when calling it using other datasets
        # def __init__(self, root, 
        #                 dataidxs=None, 
        #                 train=True, 
        #                 transform=None, 
        #                 target_transform=None, 
        #                 download=False,
        #                 user_id=0,
        #                 num_total_users=200,
        #                 poison_type="southwest"):        
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True,
                                    user_id=user_id, num_total_users=num_total_users, poison_type=poison_type,
                                    ardis_dataset_train=ardis_dataset, attack_case=attack_case)
        
        test_ds = None #dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


from torch.utils.data import TensorDataset, DataLoader
def load_poisoned_dataset(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print('load poisoned data : ', args.dataset)
    if args.dataset == "ECG5000" or args.dataset == "lightning7":
        print('load poisoned data lightning')
        # train_data = np.loadtxt(datadir + dataset +'/'+dataset + '_TRAIN.txt')
        datadir = 'D:/21120338/code/OOD_TS_FL/data/UCR/'
        poisoned_data = np.loadtxt(datadir + args.dataset +'/' + 'attack_time_series.txt')
        poisoned_data = torch.tensor(poisoned_data, dtype = torch.float)
        num_dps_poisoned_dataset = poisoned_data.shape[0]
        #poisoned_dataset = poisoned_dataset.data
        print('poisoned mnist data: ', poisoned_data.shape)

        targetted_task_test_data = np.loadtxt(datadir + args.dataset +'/' +args.dataset+ '_unseen.txt')
        targetted_task_test_data = torch.tensor(targetted_task_test_data, dtype = torch.float)
        vanilla_test_data = np.loadtxt(datadir + args.dataset +'/' +args.dataset+ '_TEST.txt')
        vanilla_test_data = torch.tensor(vanilla_test_data, dtype=torch.float)
        clean_train_data = np.loadtxt(datadir + args.dataset +'/' +args.dataset+ '_TRAIN.txt')
        clean_train_data = torch.tensor(clean_train_data, dtype=torch.float)
        #  train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), torch.from_numpy(train_labels).to(torch.float))
        # 组合数据和标签
        poisoned_dataset = TensorDataset(poisoned_data[:, 1:], poisoned_data[:, 0])
        targetted_task_test_dataset = TensorDataset(targetted_task_test_data[:, 1:], targetted_task_test_data[:, 0])
        vanilla_test_dataset = TensorDataset(vanilla_test_data[:, 1:], vanilla_test_data[:, 0])
        clean_train_dataset = TensorDataset(clean_train_data[:, 1:], clean_train_data[:, 0])

        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_dataset,
             batch_size=args.batch_size, shuffle=True, **kwargs)
        targetted_task_test_loader = torch.utils.data.DataLoader(targetted_task_test_dataset,
                                                                 batch_size=args.test_batch_size, shuffle=False,
                                                                 **kwargs)
        vanilla_test_loader = torch.utils.data.DataLoader(vanilla_test_dataset,
             batch_size=args.test_batch_size, shuffle=False, **kwargs)
        clean_train_loader = torch.utils.data.DataLoader(clean_train_dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)

    elif args.dataset in ("mnist", "emnist"):
        if args.fraction < 1:
            fraction=args.fraction  #0.1 #10
        else:
            fraction=int(args.fraction)

        with open("poisoned_dataset_fraction_{}".format(fraction), "rb") as saved_data_file:
            poisoned_dataset = torch.load(saved_data_file)
        num_dps_poisoned_dataset = poisoned_dataset.data.shape[0]
        print('poisoned mnist data: ',poisoned_dataset.data.shape)
        print('poisoned mnist data: ', poisoned_dataset.targets)
        '''prepare fashionMNIST dataset
        Transform: Compose(
               ToTensor()
               Normalize(mean=0.1307, std=0.3081)
           )
        '''
        #poisoned_dataset = poisoned_dataset.data
        poisoned_dataset.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307, std=0.3081)])
        print('poisoned mnist data: ', poisoned_dataset)
        fashion_mnist_train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307), (0.3081))
                           ]))

        fashion_mnist_test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307), (0.3081))
                           ]))
        # prepare EMNIST dataset
        emnist_train_dataset = datasets.EMNIST('./data', split="digits", train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307), (0.3081))
                           ]))

        # print(emnist_train_dataset)
        emnist_test_dataset = datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307), (0.3081))
                           ]))
        # print(emnist_test_dataset)
        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_dataset,
             batch_size=args.batch_size, shuffle=True, **kwargs)
        vanilla_test_loader = torch.utils.data.DataLoader(emnist_test_dataset,
             batch_size=args.test_batch_size, shuffle=False, **kwargs)
        targetted_task_test_loader = torch.utils.data.DataLoader(fashion_mnist_test_dataset,
             batch_size=args.test_batch_size, shuffle=False, **kwargs)
        clean_train_loader = torch.utils.data.DataLoader(emnist_train_dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)

        if args.poison_type == 'ardis':
            # load ardis test set
            with open("./data/ARDIS/ardis_test_dataset.pt", "rb") as saved_data_file:
                ardis_test_dataset = torch.load(saved_data_file)
            print('ardis', ardis_test_dataset.data.shape)
            print('ardis', ardis_test_dataset.targets)
            targetted_task_test_loader = torch.utils.data.DataLoader(ardis_test_dataset,
                 batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # emnist jieshu
    
    elif args.dataset == "cifar10":
        if args.poison_type == "southwest":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            poisoned_trainset = copy.deepcopy(trainset)
            if args.attack_case == "edge-case":
                with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
            elif args.attack_case == "normal-case" or args.attack_case == "almost-edge-case":
                with open('./saved_datasets/southwest_images_adv_p_percent_edge_case.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('./saved_datasets/southwest_images_p_percent_edge_case_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
            else:
                raise NotImplementedError("Not Matched Attack Case ...")             

            #
            logger.info("数据集大小OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
            #sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
            sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
            
            logger.info("数据集大小OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
            #sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird
            sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck
            # downsample the poisoned dataset #################
            if args.attack_case == "edge-case":
                num_sampled_poisoned_data_points = 100 # N
                samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                                num_sampled_poisoned_data_points,
                                                                replace=False)
                saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
                sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
                logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
            elif args.attack_case == "normal-case" or args.attack_case == "almost-edge-case":
                num_sampled_poisoned_data_points = 100 # N
                samped_poisoned_data_indices = np.random.choice(784,
                                                                num_sampled_poisoned_data_points,
                                                                replace=False)
            ######################################################


            # downsample the raw cifar10 dataset #################
            num_sampled_data_points = 400 # M
            samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
            poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
            logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            # keep a copy of clean data
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ########################################################


            poisoned_trainset.data = np.append(poisoned_trainset.data, saved_southwest_dataset_train, axis=0)
            poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

            logger.info("{}".format(poisoned_trainset.data.shape))
            logger.info("{}".format(poisoned_trainset.targets.shape))
            logger.info("{}".format(sum(poisoned_trainset.targets)))


            #poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_southwest_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            # vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
            # targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)

            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

        elif args.poison_type == "southwest-da":
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])

            # transform_poison = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #     AddGaussianNoise(0., 0.05),
            # ])

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])

            transform_poison = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                AddGaussianNoise(0., 0.05),
                ])            
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])

            #transform_test = transforms.Compose([
            #    transforms.ToTensor(),
            #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            #poisoned_trainset = copy.deepcopy(trainset)
            #  class CIFAR10_Poisoned(data.Dataset):
            #def __init__(self, root, clean_indices, poisoned_indices, dataidxs=None, train=True, transform_clean=None,
            #    transform_poison=None, target_transform=None, download=False):

            with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)

            with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)

            #
            logger.info("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
            sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
            
            logger.info("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
            sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck



            # downsample the poisoned dataset ###########################
            num_sampled_poisoned_data_points = 100 # N
            samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                            num_sampled_poisoned_data_points,
                                                            replace=False)
            saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
            sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
            logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
            ###############################################################


            # downsample the raw cifar10 dataset #################
            num_sampled_data_points = 400 # M
            samped_data_indices = np.random.choice(trainset.data.shape[0], num_sampled_data_points, replace=False)
            tempt_poisoned_trainset = trainset.data[samped_data_indices, :, :, :]
            tempt_poisoned_targets = np.array(trainset.targets)[samped_data_indices]
            logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            ########################################################

            poisoned_trainset = CIFAR10_Poisoned(root='./data', 
                              clean_indices=np.arange(tempt_poisoned_trainset.shape[0]), 
                              poisoned_indices=np.arange(tempt_poisoned_trainset.shape[0], tempt_poisoned_trainset.shape[0]+saved_southwest_dataset_train.shape[0]), 
                              train=True, download=True, transform_clean=transform_train,
                              transform_poison=transform_poison)
            #poisoned_trainset = CIFAR10_truncated(root='./data', dataidxs=None, train=True, transform=transform_train, download=True)
            clean_trainset = copy.deepcopy(poisoned_trainset)

            poisoned_trainset.data = np.append(tempt_poisoned_trainset, saved_southwest_dataset_train, axis=0)
            poisoned_trainset.target = np.append(tempt_poisoned_targets, sampled_targets_array_train, axis=0)

            logger.info("{}".format(poisoned_trainset.data.shape))
            logger.info("{}".format(poisoned_trainset.target.shape))


            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_southwest_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)

            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]            


        elif args.poison_type == "howto":
            """
            implementing the poisoned dataset in "How To Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)
            """
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            poisoned_trainset = copy.deepcopy(trainset)

            ##########################################################################################################################
            sampled_indices_train = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
                                    19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389]
            sampled_indices_test = [32941, 36005, 40138]
            cifar10_whole_range = np.arange(trainset.data.shape[0])
            remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_train+sampled_indices_test]
            logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(len(sampled_indices_train+sampled_indices_test)))
            saved_greencar_dataset_train = trainset.data[sampled_indices_train, :, :, :]
            #########################################################################################################################

            # downsample the raw cifar10 dataset ####################################################################################
            num_sampled_data_points = 500-len(sampled_indices_train)
            samped_data_indices = np.random.choice(remaining_indices, num_sampled_data_points, replace=False)
            poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
            logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ##########################################################################################################################

            # we load the test since in the original paper they augment the 
            with open('./saved_datasets/green_car_transformed_test.pkl', 'rb') as test_f:
                saved_greencar_dataset_test = pickle.load(test_f)

            #
            logger.info("Backdoor (Green car) train-data shape we collected: {}".format(saved_greencar_dataset_train.shape))
            sampled_targets_array_train = 2 * np.ones((saved_greencar_dataset_train.shape[0],), dtype =int) # green car -> label as bird
            
            logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
            sampled_targets_array_test = 2 * np.ones((saved_greencar_dataset_test.shape[0],), dtype =int) # green car -> label as bird/


            poisoned_trainset.data = np.append(poisoned_trainset.data, saved_greencar_dataset_train, axis=0)
            poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

            logger.info("Poisoned Trainset Shape: {}".format(poisoned_trainset.data.shape))
            logger.info("Poisoned Train Target Shape:{}".format(poisoned_trainset.targets.shape))


            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_greencar_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)
            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

        elif args.poison_type == "greencar-neo":
            """
            implementing the poisoned dataset in "How To Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)
            """
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            poisoned_trainset = copy.deepcopy(trainset)

            with open('./saved_datasets/new_green_cars_train.pkl', 'rb') as train_f:
                saved_new_green_cars_train = pickle.load(train_f)

            with open('./saved_datasets/new_green_cars_test.pkl', 'rb') as test_f:
                saved_new_green_cars_test = pickle.load(test_f)

            # we use the green cars in original cifar-10 and new collected green cars
            ##########################################################################################################################
            num_sampled_poisoned_data_points = 100 # N
            sampled_indices_green_car = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
                                    19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
            cifar10_whole_range = np.arange(trainset.data.shape[0])
            remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_green_car]
            #ori_cifar_green_cars = trainset.data[sampled_indices_green_car, :, :, :]

            samped_poisoned_data_indices = np.random.choice(saved_new_green_cars_train.shape[0],
                                                            #num_sampled_poisoned_data_points-len(sampled_indices_green_car),
                                                            num_sampled_poisoned_data_points,
                                                            replace=False)
            saved_new_green_cars_train = saved_new_green_cars_train[samped_poisoned_data_indices, :, :, :]

            #saved_greencar_dataset_train = np.append(ori_cifar_green_cars, saved_new_green_cars_train, axis=0)
            saved_greencar_dataset_train = saved_new_green_cars_train
            logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(saved_greencar_dataset_train.shape[0]))
            #########################################################################################################################

            # downsample the raw cifar10 dataset ####################################################################################
            num_sampled_data_points = 400
            samped_data_indices = np.random.choice(remaining_indices, num_sampled_data_points, replace=False)
            poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
            logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ##########################################################################################################################

            #
            logger.info("Backdoor (Green car) train-data shape we collected: {}".format(saved_greencar_dataset_train.shape))
            sampled_targets_array_train = 2 * np.ones((saved_greencar_dataset_train.shape[0],), dtype =int) # green car -> label as bird
            
            logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_new_green_cars_test.shape))
            sampled_targets_array_test = 2 * np.ones((saved_new_green_cars_test.shape[0],), dtype =int) # green car -> label as bird/


            poisoned_trainset.data = np.append(poisoned_trainset.data, saved_greencar_dataset_train, axis=0)
            poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

            logger.info("Poisoned Trainset Shape: {}".format(poisoned_trainset.data.shape))
            logger.info("Poisoned Train Target Shape:{}".format(poisoned_trainset.targets.shape))


            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_new_green_cars_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)
            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]
    # if dataset = cifar10
    return poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader
def load_ucr(path, normalize=False):
    data = np.loadtxt(path)
    # data[:, 0] -= 1
    # limit label to [0,num_classes-1]
    num_classes = len(np.unique(data[:, 0]))
    for i in range(data.shape[0]):
        if data[i, 0] < 0:# 标签小于0则重置为num_classes - 1
            data[i, 0] = num_classes - 1
    # Normalize some datasets without normalization在没有归一化的情况下使某些数据集归一化
    if normalize:
        mean = data[:, 1:].mean(axis=1, keepdims=True)
        std = data[:, 1:].std(axis=1, keepdims=True)
        data[:, 1:] = (data[:, 1:] - mean) / (std + 1e-8)
    return data # 返回归一化数据
from torch.utils.data import Dataset
class UcrDataset(Dataset):
    def __init__(self, txt_file, channel_last, normalize):
        '''
        :param txt_file: path of file
        :param channel_last
        '''
        # self.data = np.loadtxt(txt_file)
        self.data = load_ucr(txt_file, normalize)
        self.channel_last = channel_last
        if self.channel_last:
            self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1], 1])
        else:
            self.data = np.reshape(self.data, [self.data.shape[0], 1, self.data.shape[1]])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if not self.channel_last:
            return self.data[idx, :, 1:], self.data[idx, :, 0]
        else:
            return self.data[idx, 1:, :], self.data[idx, 0, :]

    def get_seq_len(self):
        if self.channel_last:
            return self.data.shape[1] - 1
        else:
            return self.data.shape[2] - 1

def seed_experiment(seed=0):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything %d",seed)

def load_UEA(dataset):
    # 读取训练和测试数据
    print(f'data/UCR/{dataset}/{dataset}_TRAIN.arff')
    train_data = loadarff(f'data/UCR/{dataset}/{dataset}_TRAIN.arff')[0]
    # print('T')
    test_data = loadarff(f'data/UCR/{dataset}/{dataset}_TEST.arff')[0]
    # file = open("./class/" + dataset + "zengqiangclass.xls", 'w')
    # file.write(dataset + '\n')
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    # 获取对应类别的下标
    def findid(y, target):#类别：标签
        index = []
        for i, lab in enumerate(y):
            if lab == target:
                index.append(i)
        return index

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    #print(train_y)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    labels = np.unique(train_y) #去除重复元素
    '''indexes = []
    for label in labels:
        index = findid(train_y, label)
        # file.write(str(len(index)) + '\n')
        print(label, len(index))
        index = random.sample(index, int(0.5*len(index))) # 随机选取80%的数据
        indexes.extend(index)'''
    #print(indexes, len(indexes))
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    '''train_X = train_X[indexes]
    train_y = train_y[indexes]'''
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    test_y = np.vectorize(transform.get)(test_y)
    # file.close()
    return train_X, train_y, test_X, test_y
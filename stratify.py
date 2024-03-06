# -*- coding:utf-8 -*-
import numpy as np
import os
from collections import Counter
import random

import matplotlib.pyplot as plt
from pyts.transformation import ShapeletTransform
from ts_attack.query_probalility import load_ucr

def labels_datasets(dataname):
    data_path = './data/UCR/' + dataname + '/' + dataname
    train_data = np.loadtxt(data_path + '_TRAIN.txt')
    test_data = np.loadtxt(data_path + '_TEST.txt')
    print('data: ', test_data.shape, train_data.shape)
    #train
    train_count = Counter(train_data[:, 0])
    print("class: ", len(train_count.keys()))
    if 0 not in train_count.keys():
        train_data[:, 0] -= 1
        num_classes = len(np.unique(train_data[:, 0]))
        for i in range(train_data.shape[0]):
            if train_data[i, 0] < 0:  # 标签小于0则重置为num_classes - 1
                train_data[i, 0] = num_classes - 1
    # test
    test_count = Counter(test_data[:, 0])
    print("class: ", len(test_count.keys()))
    if 0 not in test_count.keys():
        test_data[:, 0] -= 1
        num_classes = len(np.unique(test_data[:, 0]))
        for i in range(test_data.shape[0]):
            if test_data[i, 0] < 0:  # 标签小于0则重置为num_classes - 1
                test_data[i, 0] = num_classes - 1
    np.savetxt(data_path + '_TEST.txt', test_data)
    np.savetxt(data_path + '_TRAIN.txt', train_data)
    train_data = np.loadtxt(data_path + '_TRAIN.txt')
    test_data = np.loadtxt(data_path + '_TEST.txt')
    print('data after:test,train ', test_data.shape, train_data.shape)

def shapelet_transform(run_tag):
    # Shapelet transformation
    st = ShapeletTransform(n_shapelets=5, window_sizes=[0.1, 0.2, 0.3, 0.4
                                                        ], sort=True, verbose=1, n_jobs=1)
    path = './data/UCR/' + run_tag + '/' + run_tag + '_TRAIN.txt'
    data = load_ucr(path)
    X = data[:, 1:]
    mask = np.isnan(X)
    X[mask] = 0
    y = data[:, 0]
    print('shapelet transfor')
    X_new = st.fit_transform(X, y)
    print('save shapelet pos')
    file = open('./shapelet_pos/' + run_tag + '_shapelet_pos.txt', 'w+')
    for i, index in enumerate(st.indices_):
        idx, start, end = index
        file.write(run_tag + ' ')
        file.write(str(idx) + ' ')
        file.write(str(start) + ' ')
        file.write(str(end) + '\n')
    file.close()
    # Visualize the most discriminative shapelets
    plt.figure(figsize=(6, 4))
    for i, index in enumerate(st.indices_):
        idx, start, end = index
        plt.plot(X[idx], color='C{}'.format(i),
                 label='Sample {}'.format(idx))
        plt.plot(np.arange(start, end), X[idx, start:end],
                 lw=5, color='C{}'.format(i))

    plt.xlabel('Time', fontsize=12)
    plt.title('The five more discriminative shapelets', fontsize=14)
    plt.legend(loc='best', fontsize=8)
    plt.savefig('./shapelet_fig/' + run_tag + '_shapelet.pdf')
    # plt.show()


def stratify_attack_data(dataname):
    data_path = './data/UCR/' + dataname + '/' + dataname
    data = np.loadtxt(data_path + '_TRAIN.txt')
    #data = np.loadtxt(data_path + '_TEST.txt')
    size = data.shape[0]
    test_count = Counter(data[:, 0])
    print(test_count)
    def findid(y, target):  # 类别：标签
        index = []
        for i, lab in enumerate(y):
            if lab == target:
                index.append(i)
        return index
    index = list(np.arange(size))
    #index = random.shuffle(list(np.arange(size)))

    # file.write(str(len(index)) + '\n')
    # print(label, len(index))
    # ind_att = random.sample(index, int(0.05*len(index))) #
    num_att = int(0.05 * data.shape[0])
    ind_att = random.sample(index, num_att)  #
    print(ind_att)
    attack_data = data[ind_att]
    no_attack = np.delete(np.arange(size), ind_att)
    eval_data = data[no_attack]
    '''
    np.savetxt("a.txt", a, fmt="%d", delimiter=",") #改为保存为整数，以逗号分隔  
    np.loadtxt("a.txt",delimiter=",") # 读入的时候也需要指定逗号分隔'''

    np.savetxt(data_path+ '_no_attack.txt', eval_data)
    np.savetxt(data_path + '_attack.txt', attack_data)
    print(type(eval_data))

    # eval_data = np.loadtxt(data_path + '_eval.txt')
    # unseen_data = np.loadtxt(data_path + '_unseen.txt')
    print('data: ', data.shape)
    print('train: attack data: ', eval_data.shape, attack_data.shape)

if __name__ == '__main__':
    '''name = 'ECG5000','ShapesAll','SwedishLeaf' 'StarLightCurves' 'TwoPatterns''MelbournePedestrian'''
    names = ['MelbournePedestrian']

    for name in names:
        print('######## Start %s stratify #####' % name)
        stratify_attack_data(name)
        #shapelet_transform(name)
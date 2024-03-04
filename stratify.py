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

def stratify_datasets(dataname):
    # 用于把test数据分层采样为eval和unseen数据集
    data_path = './data/UCR/' + dataname + '/' + dataname
    train_data = np.loadtxt(data_path + '_TRAIN.txt')
    test_data = np.loadtxt(data_path + '_TEST.txt')
    data = np.vstack((train_data, test_data))
    print('data: ', test_data.shape, train_data.shape, data.shape)
    size = data.shape[0]
    test_count = Counter(data[:, 0])
    print("class: ", len(test_count.keys()))
    # 获取对应类别的下标
    def findid(y, target):  # 类别：标签
        index = []
        for i, lab in enumerate(y):
            if lab == target:
                index.append(i)
        return index
    # 找出每一类的下标
    indexes = []
    for label in test_count.keys():
        index = findid(data[:, 0], label)
        # file.write(str(len(index)) + '\n')
        # print(label, len(index))
        index = random.sample(index, int(0.3*len(index))) # 随机选取40%的数据测试
        indexes.extend(index)
    if 0 not in test_count.keys():
        data[:, 0] -= 1
        print('limit label to [0,num_classes-1]')
        num_classes = len(np.unique(data[:, 0]))
        for i in range(data.shape[0]):
            if data[i, 0] < 0:  # 标签小于0则重置为num_classes - 1
                data[i, 0] = num_classes - 1
    eval_data = data[indexes]
    no_indexes = np.delete(np.arange(size), indexes)
    # print(type(indexes))
    # print(type(no_indexes))
    traindataset = data[no_indexes]
    '''
    np.savetxt("a.txt", a, fmt="%d", delimiter=",") #改为保存为整数，以逗号分隔  
    np.loadtxt("a.txt",delimiter=",") # 读入的时候也需要指定逗号分隔'''

    np.savetxt(data_path+ '_TEST.txt', eval_data)
    np.savetxt(data_path + '_TRAIN.txt', traindataset)
    # print(type(eval_data))

    # eval_data = np.loadtxt(data_path + '_eval.txt')
    # unseen_data = np.loadtxt(data_path + '_unseen.txt')

    print('testdata: ', data.shape)

    print('test class num: ', test_count)
    train_data = np.loadtxt(data_path + '_TRAIN.txt')
    test_data = np.loadtxt(data_path + '_TEST.txt')
    print('data after:test,train ', test_data.shape, train_data.shape)


def stratify_attack_data(dataname):
    # 用于生产某一类attack数据集
    data_path = './data/UCR/' + dataname + '/' + dataname
    data = np.loadtxt(data_path + '_TRAIN.txt')
    #data = np.loadtxt(data_path + '_TEST.txt')
    size = data.shape[0]
    test_count = Counter(data[:, 0])
    print(test_count)
    # 获取对应类别的下标
    def findid(y, target):  # 类别：标签
        index = []
        for i, lab in enumerate(y):
            if lab == target:
                index.append(i)
        return index
    # 找出每一类的下标
    label = 0
    index = findid(data[:, 0], label)
    # file.write(str(len(index)) + '\n')
    # print(label, len(index))
    # ind_att = random.sample(index, int(0.05*len(index))) #
    num_att = int(min(0.5*len(index), 0.03 * data.shape[0]))
    ind_att = random.sample(index, num_att)  #
    attack_data = data[ind_att]
    no_attack = np.delete(np.arange(size), ind_att)
    print(type(ind_att))
    print(type(no_attack))
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
    # ECG, Sensor, Device, Spectro
    '''name = [  # ECG+Sensor:24
        'Car', 'ChlorineConcentration', 'CinCECGTorso',
        'Earthquakes', 'ECG5000', 'ECG200', 'ECGFiveDays',
        'FordA', 'FordB',
        'InsectWingbeatSound', 'ItalyPowerDemand',
        'Lightning2', 'Lightning7',
        'MoteStrain',
        'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
        'Plane', 'Phoneme',
        'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
        'Trace', 'TwoLeadECG',
        'Wafer',
        'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',  # ECG+Sensor+HRM:18
        'FreezerRegularTrain', 'FreezerSmallTrain',
        'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
        'ShakeGestureWiimoteZ',
        'Fungi',
        'Crop''ElectricDevices''KeplerLightCurves''NonInvasiveFetalECGThorax1'
        'GesturePebbleZ1', 'GesturePebbleZ2',
        'DodgerLoopDay', 'DodgerLoopWeekend', 'DodgerLoopGame','UWaveGestureLibraryX'
        'EOGHorizontalSignal', 'EOGVerticalSignal']
        'ShapesAll','SwedishLeaf' 'StarLightCurves' 'TwoPatterns''MelbournePedestrian''MixedShapesSmallTrain'
        '''
    names = ['ShapesAll']

    for name in names:
        print('######## Start %s stratify stratify #####' % name)
        #stratify_datasets(name)
        #labels_datasets(name)
        #stratify_attack_data(name)
        print('######## Start %s shapelet_transform #####' % name)
        shapelet_transform(name)
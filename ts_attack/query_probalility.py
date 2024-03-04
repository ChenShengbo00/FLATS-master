'''
This script use pre-trained model(e.g. FCN)
as the target model. We can query the probability
from it to decide attacks whether efficient or not.
'''
import numpy as np
import torch
import torch.nn as nn


def load_ucr(path, normalize=False):
    data = np.loadtxt(path)
    data[:, 0] -= 1
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


def query_one(device, model, sample_ts, attack_ts, labels, n_class, target_class=-1, verbose=False, cuda=False):
    # device = torch.device("cuda:0" if cuda else "cpu")
    ts = torch.tensor(attack_ts)
    # ts = torch.from_numpy(attack_ts).float()
    # data_path = 'data/' + run_tag + '/' + run_tag + '_unseen.txt'
    # test_data = load_ucr(path=data_path, normalize=normalize)
    # test_data = torch.from_numpy(test_data)
    # n_class = torch.unique(Y).size(0) #去除重复元素
    test_one = torch.tensor(sample_ts)
    # 取出测试样例的特征和标签
    X = test_one.float()
    y = labels.long()
    y = y.to(device)

    real_label = y

    if target_class != -1:
        y = target_class #攻击标签
    # 用于攻击的序列
    ts = ts.to(device)
    X = X.to(device) # 原始序列
    # model_path = 'model_checkpoints/' + run_tag + '/pre_trained.pth'
    # model = torch.load(model_path, map_location='cpu')
    with torch.no_grad():

        model.eval()
        softmax = nn.Softmax(dim=-1)
        out = model(X)
        prob_vector = softmax(out) # 表示向量1
        prob = prob_vector.view(n_class)[y].item()
        out2 = model(ts)
        prob_vector2 = softmax(out2) # 分类概率向量
        prob2 = prob_vector2.view(n_class)[y].item()
        if verbose:
            print('Target_Class：', target_class)
            print('Prior Confidence of the sample is  %.4f ' % (prob))
    # 返回输入序列的分类概率，分为每一类的概率，原始数据的分类概率。真实标签
    return prob2, prob_vector2, prob, prob_vector


if __name__ == '__main__':
    query_one('ECG5000', 2)

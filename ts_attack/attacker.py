import warnings
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from datetime import datetime
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from ts_attack.query_probalility import  query_one, load_ucr
warnings.filterwarnings('ignore')


def merge(intervals):
    """
    Merge shapelet interval
    :type intervals: List[List[int]]
    :rtype: List[List[int]]
    :return: the merged shapelet intervals(2d-list)
    """
    #print(intervals)
    if len(intervals) == 0:
        return []

    res = []
    intervals = list(sorted(intervals))
    # 初始化两个端点
    low = intervals[0][0]
    high = intervals[0][1]
    for i in range(1, len(intervals)):
        if high >= intervals[i][0]:
            if high < intervals[i][1]:
                high = intervals[i][1]
        else:
            res.append([low, high])
            low = intervals[i][0]
            high = intervals[i][1]
    print('res', res)
    res.append([low, high])
    return res


def get_interval(run_tag, topk):
    '''
    :param topk: the k shapelets
    :param run_tag: e.g. ECG200
    :return: shapelet interval  after merging
    '''
    shaplet_pos = np.loadtxt('./shapelet_pos/' + run_tag + '_shapelet_pos.txt', usecols=(2, 3))
    shaplet_pos = shaplet_pos[:topk]
    shaplet_pos = shaplet_pos.tolist()
    # 合并
    return merge(shaplet_pos)


def get_magnitude(run_tag, factor, normalize):
    '''
    :param run_tag:
    :param factor:
    :return: Perturbed Magnitude
    datadir = './data/UCR/'
    poisoned_data = np.loadtxt(datadir + args.dataset + '/' + args.dataset + '_attack.txt')
    '''
    data = load_ucr('./data/UCR/' + run_tag + '/' + run_tag + '_attack.txt', normalize=normalize)
    
    mask = np.isnan(data)
    data[mask] = 0
    X = data[:, 1:]

    max_magnitude = X.max(1)
    min_magnitude = X.min(1)
    mean_magnitude = np.mean(max_magnitude - min_magnitude)

    perturbed_mag = mean_magnitude * factor

    return perturbed_mag


def get_minmax(run_tag, factor, normalize):
    '''
    :param run_tag:
    :param factor:
    :return: Perturbed Magnitude
    datadir = './data/UCR/'
    poisoned_data = np.loadtxt(datadir + args.dataset + '/' + args.dataset + '_attack.txt')
    '''
    data = load_ucr('./data/UCR/' + run_tag + '/' + run_tag + '_attack.txt',
                    normalize=normalize)

    mask = np.isnan(data)
    data[mask] = 0
    X = data[:, 1:]

    max_magnitude = X.max(1)
    min_magnitude = X.min(1)

    return min_magnitude.min(), max_magnitude.max()

class Attacker:
    def __init__(self,model, device, optimizer, criterion, dataset, top_k, n_class, cuda,  e):
        # 数据名称，采用顶部K形状，最大值为5，FCNResNet，GPU，时间序列形状，规范化
        # model, device, optimizer, epoch, log_interval, criterion, data=train_loader, top_k=3, e=1499
        self.run_tag = dataset
        self.top_k = top_k
        self.n_class = n_class
        # self.model_type = model_type
        self.cuda = cuda
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.intervals = get_interval(self.run_tag, self.top_k)
        # self.normalize = normalize
        self.e = e

    def perturb_ts(self, perturbations, ts):
        '''扰动（位置，e序列）原始时间序列
        :param perturbations:formalized as a tuple（x,e),x(int) is the x-coordinate，e(float) is the epsilon,e.g.,(2,0.01)
        :param ts: time series
        :return: perturbed ts
        '''
        # first we copy a ts
        ts_tmp = np.copy(ts.cpu())
        coordinate = 0
        for interval in self.intervals:
            # 在每一段时间表示上加上扰动
            for i in range(int(interval[0]), int(interval[1])):
                ts_tmp[i] += perturbations[coordinate]
                coordinate += 1
        return ts_tmp

    def plot_per(self, perturbations, ts, target_class, attack_ts, prior_probs, attack_probs, factor):

        # Obtain the perturbed ts
        ts_tmp = np.copy(ts.cpu())
        #ts_perturbed = self.perturb_ts(perturbations=perturbations, ts=ts)
        # Start to plot
        plt.cla()
        plt.figure(figsize=(6, 4))
        plt.plot(ts_tmp, color='b', label='Original %.2f' % prior_probs)
        plt.plot(attack_ts, color='r', label='Perturbed %.2f' % attack_probs)
        plt.xlabel('Time', fontsize=12)

        if target_class == -1:
            plt.title('Untargeted: Sample %d, eps_factor=%.3f' %
                      (1, factor), fontsize=14)
        else:
            plt.title('Targeted(%d): Sample %d, eps_factor=%.3f' %
                      (target_class, 1, factor), fontsize=14)
        plt.legend(loc='upper right', fontsize=8)
        save_dir = './ts_attack/flts_attackfig/' + self.run_tag
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + '/'+str(factor) + '_' + str(self.top_k)
                    + self.run_tag + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.png')
        # plt.show()

    def fitness(self, perturbations, sample_ts, real_label, queries, target_class=-1):

        queries[0] += 1
        ts_perturbed = self.perturb_ts(perturbations, sample_ts)
        # 返回ts_perturb序列的分类概率
        # query_one(model, sample_ts, attack_ts, labels, n_class, target_class=-1, verbose=False, cuda=False):
        prob, _, _, _ = query_one(device = self.device, model=self.model, sample_ts=sample_ts,
                                            attack_ts=ts_perturbed,
                                            labels=real_label, n_class=self.n_class,
                                            target_class=target_class,verbose=False,
                                            cuda=self.cuda)

        if target_class != -1:
            prob = 1 - prob

        return prob  # The fitness function is to minimize the fitness value健身功能是最大程度地减少健身价值

    def attack_success(self, perturbations, sample_ts, real_label, iterations, target_class=-1, verbose=False):

        iterations[0] += 1
        print('attack success The %d iteration' % iterations[0])
        ts_perturbed = self.perturb_ts(perturbations, sample_ts) #返回扰动的序列
        # Obtain the perturbed probability vector and the prior probability vector
        # 返回输入序列的分类概率，原始数据的分类概率。真实标签
        # query_one(model, sample_ts, attack_ts, labels, n_class, target_class=-1, verbose=False, cuda=False):
        prob, prob_vector, prior_prob, prior_prob_vec = query_one(device = self.device, model=self.model, sample_ts=sample_ts,
                                                                              attack_ts=ts_perturbed,labels=real_label,
                                                                  n_class=self.n_class,
                                                                              target_class=target_class,
                                                                              verbose=verbose, cuda=self.cuda)

        predict_class = torch.argmax(prob_vector) #输入序列的分类
        prior_class = torch.argmax(prior_prob_vec) #原始序列的分类

        # Conditions for early termination(empirical-based estimation), leading to save the attacking time
        # But it may judge incorrectly that this may decrease the success rate of the attack.
        if (iterations[0] > 5 and prob > 0.99) or (iterations[0] > 20 and prob > 0.9):
            print('The sample sample is not expected to successfully attack.')
            return True

        if prior_class != real_label:
            print('The sample cannot be classified correctly, no need to attack')
            return True

        if prior_class == target_class:
            print(
                'The true label of sample equals to target label, no need to attack')
            return True

        if verbose:
            print('The Confidence of current iteration: %.4f' % prob)
            print('###################################################')

        # The criterion of attacking successfully:
        # Untargeted attack: predicted label is not equal to the original label.
        # Targeted attack: predicted label is equal to the target label.
        if ((target_class == -1 and predict_class != prior_class) or
                (target_class != -1 and predict_class == target_class)):
            print('################## Attack Successfully! ##################')
            return True

    def attack(self, sample_ts, reallabels, target_class=-1, factor=0.04, max_iteration=60, popsize=200, verbose=True):
        # 原始时间序列
        ori_ts = sample_ts
        real_label =reallabels
        # (model, sample_ts, attack_ts, labels, n_class, target_class=-1, verbose=False, cuda=False):
        attacked_probs, attacked_vec, prior_probs, prior_vec = query_one(device=self.device, model=self.model, sample_ts=sample_ts,
                                                                         attack_ts=ori_ts,
                                                                         labels=real_label, n_class=self.n_class,
                                                                        target_class=target_class,verbose=False,
                                                                        cuda=self.cuda)
        prior_class = torch.argmax(prior_vec)
        if prior_class != real_label:
            print('The sample cannot be classified correctly, no need to attack')
            return ori_ts, [prior_probs, attacked_probs, 0, 0, 0, 0, 0, 'WrongSample']

        steps_count = 0  # count the number of coordinates 计算坐标的数量

        # Get the maximum perturbed magnitude 获取最大扰动幅度
        # perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=True)
        perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=False)

        bounds = []
        # 序列shapelet
        for interval in self.intervals:
            steps_count += int(interval[1]) - int(interval[0]) # 计算改变点的数量
            for i in range(int(interval[0]), int(interval[1])):
                bounds.append((-1 * perturbed_magnitude, perturbed_magnitude))
        popmul = max(1, popsize // len(bounds))
        # Record of the number of iterations 迭代次数的记录
        iterations = [0]
        queries = [0]

        def fitness_fn(perturbations):
            # fitness(self, perturbations, sample_ts, real_label, queries, target_class=-1):
            return self.fitness(perturbations=perturbations, sample_ts=ori_ts, queries=queries,
                                real_label=real_label, target_class=target_class)

        def callback_fn(x, convergence):
            # attack_success(self, perturbations, sample_ts, real_label, iterations, target_class=-1, verbose=True):
            return self.attack_success(perturbations=x,
                                       sample_ts=sample_ts,
                                       real_label=real_label,
                                       iterations=iterations,
                                       target_class=target_class,
                                       verbose=verbose)
        '''
        找到多元函数的全局最小值。差异进化本质上是随机的（不使用梯度方法）找到最低限度，可以搜索大部分候选人空间，
        但通常需要大量的功能评估常规的基于梯度的技术,返回`x'解决方案数组。
        result.x, result.fun, result.nfev'''
        attack_result = differential_evolution(func=fitness_fn, bounds=bounds
                                               , maxiter=max_iteration, popsize=popmul
                                               , recombination=0.7, callback=callback_fn,
                                               atol=-1, polish=False)
        attack_ts = self.perturb_ts(attack_result.x, ori_ts)
        
        mse = mean_squared_error(ori_ts.cpu(), attack_ts)
        # 测试攻击
        attacked_probs, attacked_vec, prior_probs, prior_vec = query_one(device=self.device, model=self.model, sample_ts=sample_ts,
                                                                         attack_ts=attack_ts,
                                                                         labels=real_label, n_class=self.n_class,
                                                                        target_class=target_class,verbose=False,
                                                                        cuda=self.cuda)

        predicted_class = torch.argmax(attacked_vec)
        prior_class = torch.argmax(prior_vec)
        # 原始预测类别不等于真实类别
        if prior_class != real_label:
            success = 'WrongSample'
        # 原始预测类别等于目标类别
        elif prior_class == target_class:
            success = 'NoNeedAttack'
        # 需要攻击
        else:
            if (predicted_class.item() != prior_class.item() and target_class == -1) \
                    or (predicted_class.item() == target_class and target_class != -1):
                # 攻击成功
                success = 'Success'
            else:
                success = 'Fail'

        if success == 'Success':
            try:
                self.plot_per(perturbations=attack_result.x, ts=ori_ts, target_class=target_class,
                          attack_ts=attack_ts, prior_probs=prior_probs, attack_probs=attacked_probs, factor=factor)
            except Exception as e:
                print("erroe: ", e)
        return attack_ts, [prior_probs, attacked_probs, prior_class.item(),
                           predicted_class.item(), queries[0], mse, iterations[0], success]

'''
随机shapelet区间扰动'''
class AttackerRandShape:
    def __init__(self, model, device, optimizer, criterion, dataset, top_k, n_class, cuda, e):
        # 数据名称，采用顶部K形状，最大值为5，FCNResNet，GPU，时间序列形状，规范化
        # model, device, optimizer, epoch, log_interval, criterion, data=train_loader, top_k=3, e=1499
        self.run_tag = dataset
        self.top_k = top_k
        self.n_class = n_class
        # self.model_type = model_type
        self.cuda = cuda
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.intervals = get_interval(self.run_tag, self.top_k)
        # self.normalize = normalize
        self.e = e

    def perturb_ts(self, perturbations, ts):
        '''扰动（位置，e序列）原始时间序列
        :param perturbations:formalized as a tuple（x,e),x(int) is the x-coordinate，e(float) is the epsilon,e.g.,(2,0.01)
        :param ts: time series
        :return: perturbed ts
        '''
        # first we copy a ts
        ts_tmp = np.copy(ts.cpu())
        coordinate = 0
        for interval in self.intervals:
            # 在每一段时间表示上加上扰动
            for i in range(int(interval[0]), int(interval[1])):
                ts_tmp[i] += perturbations[coordinate]
                coordinate += 1
        return ts_tmp

    def plot_per(self, perturbations, ts, target_class, attack_ts, prior_probs, attack_probs, factor):

        # Obtain the perturbed ts
        ts_tmp = np.copy(ts.cpu())
        # ts_perturbed = self.perturb_ts(perturbations=perturbations, ts=ts)
        # Start to plot
        plt.cla()
        plt.figure(figsize=(6, 4))
        plt.plot(ts_tmp, color='b', label='Original %.2f' % prior_probs)
        plt.plot(attack_ts, color='r', label='Perturbed %.2f' % attack_probs)
        plt.xlabel('Time', fontsize=12)

        if target_class == -1:
            plt.title('Untargeted: Sample %d, eps_factor=%.3f' %
                      (1, factor), fontsize=14)
        else:
            plt.title('Targeted(%d): Sample %d, eps_factor=%.3f' %
                      (target_class, 1, factor), fontsize=14)
        plt.legend(loc='upper right', fontsize=8)
        save_dir = './ts_attack/flts_attackfig/' + self.run_tag + 'ShapeRandom'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + '/' + str(factor) + '_' + str(self.top_k)
                    + self.run_tag + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.png')
        # plt.show()

    def attack(self, sample_ts, reallabels, target_class=-1, factor=0.04, max_iteration=60, popsize=200, verbose=True):
        # 原始时间序列
        ori_ts = sample_ts
        real_label = reallabels
        # (model, sample_ts, attack_ts, labels, n_class, target_class=-1, verbose=False, cuda=False):
        attacked_probs, attacked_vec, prior_probs, prior_vec = query_one(device=self.device, model=self.model,
                                                                         sample_ts=sample_ts,
                                                                         attack_ts=ori_ts,
                                                                         labels=real_label, n_class=self.n_class,
                                                                         target_class=target_class, verbose=False,
                                                                         cuda=self.cuda)
        prior_class = torch.argmax(prior_vec)
        if prior_class != real_label:
            print('The sample cannot be classified correctly, no need to attack')
            return ori_ts, [prior_probs, attacked_probs, 0, 0, 0, 0, 0, 'WrongSample']

        steps_count = 0  # count the number of coordinates 计算坐标的数量

        # Get the maximum perturbed magnitude 获取最大扰动幅度
        # perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=True)
        perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=False)

        bounds = []
        # 序列shapelet
        for interval in self.intervals:
            steps_count += int(interval[1]) - int(interval[0])  # 计算改变点的数量
            for i in range(int(interval[0]), int(interval[1])):
                bounds.append(perturbed_magnitude * (2*np.random.random() - 1))
        iterations = [0]
        queries = [0]
        attack_ts = self.perturb_ts(bounds, ori_ts)

        mse = mean_squared_error(ori_ts.cpu(), attack_ts)
        # 测试攻击
        attacked_probs, attacked_vec, prior_probs, prior_vec = query_one(device=self.device, model=self.model,
                                                                         sample_ts=sample_ts,
                                                                         attack_ts=attack_ts,
                                                                         labels=real_label, n_class=self.n_class,
                                                                         target_class=target_class, verbose=False,
                                                                         cuda=self.cuda)

        predicted_class = torch.argmax(attacked_vec)
        prior_class = torch.argmax(prior_vec)
        # 原始预测类别不等于真实类别
        if prior_class != real_label:
            success = 'WrongSample'
        # 原始预测类别等于目标类别
        elif prior_class == target_class:
            success = 'NoNeedAttack'
        # 需要攻击
        else:
            if (predicted_class.item() != prior_class.item() and target_class == -1) \
                    or (predicted_class.item() == target_class and target_class != -1):
                # 攻击成功
                success = 'Success'
            else:
                success = 'Fail'

        if success == 'Success':
            try:
                self.plot_per(perturbations=bounds, ts=ori_ts, target_class=target_class,
                          attack_ts=attack_ts, prior_probs=prior_probs, attack_probs=attacked_probs, factor=factor)
            except Exception as e:
                print("erroe: ", e)

        return attack_ts, [prior_probs, attacked_probs, prior_class.item(),
                           predicted_class.item(), queries[0], mse, iterations[0], success]

class AttackerRandAll:
    def __init__(self, model, device, optimizer, criterion, dataset, top_k, n_class, cuda, e):
        # 数据名称，采用顶部K形状，最大值为5，FCNResNet，GPU，时间序列形状，规范化
        # model, device, optimizer, epoch, log_interval, criterion, data=train_loader, top_k=3, e=1499
        self.run_tag = dataset
        self.top_k = top_k
        self.n_class = n_class
        # self.model_type = model_type
        self.cuda = cuda
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.intervals = get_interval(self.run_tag, self.top_k)
        # self.normalize = normalize
        self.e = e

    def perturb_ts(self, perturbed_magnitude, ts):
        '''扰动（位置，e序列）原始时间序列
        :param perturbations:formalized as a tuple（x,e),x(int) is the x-coordinate，e(float) is the epsilon,e.g.,(2,0.01)
        :param ts: time series
        :return: perturbed ts
        '''
        # first we copy a ts
        ts_tmp = np.copy(ts.cpu())
        # 在每一段时间表示上加上扰动
        for i in range(0, len(ts_tmp)):
            ts_tmp[i] += perturbed_magnitude * (2*np.random.random() - 1)
        return ts_tmp

    def plot_per(self, perturbations, ts, target_class, attack_ts, prior_probs, attack_probs, factor):

        # Obtain the perturbed ts
        ts_tmp = np.copy(ts.cpu())
        # ts_perturbed = self.perturb_ts(perturbations=perturbations, ts=ts)
        # Start to plot
        plt.cla()
        plt.figure(figsize=(6, 4))
        plt.plot(ts_tmp, color='b', label='Original %.2f' % prior_probs)
        plt.plot(attack_ts, color='r', label='Perturbed %.2f' % attack_probs)
        plt.xlabel('Time', fontsize=12)

        if target_class == -1:
            plt.title('Untargeted: Sample %d, eps_factor=%.3f' %
                      (1, factor), fontsize=14)
        else:
            plt.title('Targeted(%d): Sample %d, eps_factor=%.3f' %
                      (target_class, 1, factor), fontsize=14)
        plt.legend(loc='upper right', fontsize=8)
        save_dir = './ts_attack/flts_attackfig/' + self.run_tag + 'overAllRandom'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + '/' + str(factor) + '_'
                    + self.run_tag + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.png')
        # plt.show()

    def attack(self, sample_ts, reallabels, target_class=-1, factor=0.04, max_iteration=60, popsize=200, verbose=True):
        # 原始时间序列
        ori_ts = sample_ts
        real_label = reallabels
        # (model, sample_ts, attack_ts, labels, n_class, target_class=-1, verbose=False, cuda=False):
        attacked_probs, attacked_vec, prior_probs, prior_vec = query_one(device=self.device, model=self.model,
                                                                         sample_ts=sample_ts,
                                                                         attack_ts=ori_ts,
                                                                         labels=real_label, n_class=self.n_class,
                                                                         target_class=target_class, verbose=False,
                                                                         cuda=self.cuda)
        prior_class = torch.argmax(prior_vec)
        if prior_class != real_label:
            print('The sample cannot be classified correctly, no need to attack')
            return ori_ts, [prior_probs, attacked_probs, 0, 0, 0, 0, 0, 'WrongSample']

        steps_count = 0  # count the number of coordinates 计算坐标的数量

        # Get the maximum perturbed magnitude 获取最大扰动幅度
        # perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=True)
        perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=False)

        iterations = [0]
        queries = [0]
        attack_ts = self.perturb_ts(perturbed_magnitude, ori_ts)

        mse = mean_squared_error(ori_ts.cpu(), attack_ts)
        # 测试攻击
        attacked_probs, attacked_vec, prior_probs, prior_vec = query_one(device=self.device, model=self.model,
                                                                         sample_ts=sample_ts,
                                                                         attack_ts=attack_ts,
                                                                         labels=real_label, n_class=self.n_class,
                                                                         target_class=target_class, verbose=False,
                                                                         cuda=self.cuda)

        predicted_class = torch.argmax(attacked_vec)
        prior_class = torch.argmax(prior_vec)
        # 原始预测类别不等于真实类别
        if prior_class != real_label:
            success = 'WrongSample'
        # 原始预测类别等于目标类别
        elif prior_class == target_class:
            success = 'NoNeedAttack'
        # 需要攻击
        else:
            if (predicted_class.item() != prior_class.item() and target_class == -1) \
                    or (predicted_class.item() == target_class and target_class != -1):
                # 攻击成功
                success = 'Success'
            else:
                success = 'Fail'

        if success == 'Success':
            try:
                self.plot_per(perturbations=perturbed_magnitude, ts=ori_ts, target_class=target_class,
                          attack_ts=attack_ts, prior_probs=prior_probs, attack_probs=attacked_probs, factor=factor)
            except Exception as e:
                print("erroe: ", e)

        return attack_ts, [prior_probs, attacked_probs, prior_class.item(),
                           predicted_class.item(), queries[0], mse, iterations[0], success]

class AttackerOnepoint:
    def __init__(self, model, device, optimizer, criterion, dataset, top_k, n_class, cuda, e):
        # 数据名称，采用顶部K形状，最大值为5，FCNResNet，GPU，时间序列形状，规范化
        # model, device, optimizer, epoch, log_interval, criterion, data=train_loader, top_k=3, e=1499
        self.run_tag = dataset
        self.n_class = n_class
        self.cuda = cuda
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.e = e

    def plot_per(self, ts, target_class, attack_ts, prior_probs, attack_probs, factor):

        # Obtain the perturbed ts
        ts_tmp = np.copy(ts.cpu())
        # ts_perturbed = self.perturb_ts(perturbations=perturbations, ts=ts)
        # Start to plot
        plt.cla()
        plt.figure(figsize=(6, 4))
        plt.plot(ts_tmp, color='b', label='Original %.2f' % prior_probs)
        plt.plot(attack_ts, color='r', label='Perturbed %.2f' % attack_probs)
        plt.xlabel('Time', fontsize=12)

        if target_class == -1:
            plt.title('Untargeted: Sample %d, eps_factor=%.3f' %
                      (1, factor), fontsize=14)
        else:
            plt.title('Targeted(%d): Sample %d, eps_factor=%.3f' %
                      (target_class, 1, factor), fontsize=14)
        plt.legend(loc='upper right', fontsize=8)
        save_dir = './ts_attack/flts_attackfig/' + self.run_tag + 'Onepoint'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + '/' + self.run_tag + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.png')
        # plt.show()

    def attack(self, sample_ts, reallabels, target_class=-1, factor=0.04, max_iteration=60, popsize=200, verbose=True):
        # 原始时间序列
        ori_ts = sample_ts
        real_label = reallabels
        # (model, sample_ts, attack_ts, labels, n_class, target_class=-1, verbose=False, cuda=False):
        attacked_probs, attacked_vec, prior_probs, prior_vec = query_one(device=self.device, model=self.model,
                                                                         sample_ts=sample_ts,
                                                                         attack_ts=ori_ts,
                                                                         labels=real_label, n_class=self.n_class,
                                                                         target_class=target_class, verbose=False,
                                                                         cuda=self.cuda)
        prior_class = torch.argmax(prior_vec)
        if prior_class != real_label:
            print('The sample cannot be classified correctly, no need to attack')
            return ori_ts, [prior_probs, attacked_probs, 0, 0, 0, 0, 0, 'WrongSample']

        steps_count = 0  # count the number of coordinates 计算坐标的数量

        # Get the maximum perturbed magnitude 获取最大扰动幅度
        # perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=True)
        min_data, max_data = get_minmax(self.run_tag, factor, normalize=False)

        iterations = [0]
        queries = [0]
        attack_ts = np.copy(ori_ts.cpu())
        index = np.random.randint(0, len(ori_ts))
        attack_ts[index] = min_data

        mse = mean_squared_error(ori_ts.cpu(), attack_ts)
        # 测试攻击
        attacked_probs, attacked_vec, prior_probs, prior_vec = query_one(device=self.device, model=self.model,
                                                                         sample_ts=sample_ts,
                                                                         attack_ts=attack_ts,
                                                                         labels=real_label, n_class=self.n_class,
                                                                         target_class=target_class, verbose=False,
                                                                         cuda=self.cuda)

        predicted_class = torch.argmax(attacked_vec)
        prior_class = torch.argmax(prior_vec)
        # 原始预测类别不等于真实类别
        if prior_class != real_label:
            success = 'WrongSample'
        # 原始预测类别等于目标类别
        elif prior_class == target_class:
            success = 'NoNeedAttack'
        # 需要攻击
        else:
            if (predicted_class.item() != prior_class.item() and target_class == -1) \
                    or (predicted_class.item() == target_class and target_class != -1):
                # 攻击成功
                success = 'Success'
            else:
                success = 'Fail'

        if success == 'Success':
            try:
                self.plot_per(ts=ori_ts, target_class=target_class,
                          attack_ts=attack_ts, prior_probs=prior_probs, attack_probs=attacked_probs, factor=factor)
            except Exception as e:
                print("erroe: ", e)

        return attack_ts, [prior_probs, attacked_probs, prior_class.item(),
                           predicted_class.item(), queries[0], mse, iterations[0], success]

import numpy as np
from matplotlib import pyplot as pl
from matplotlib import pyplot as plt
import time
from datetime import datetime
def drawacc(datadir, mainacc, raw_task_acc, backdoor, dataset='defalt'):
    # main_task_acc, raw_task_acc, trainacc, backdoor_task_acc
    '''main_task_acc
    backdoor_acc
    raw_task_acc'''
    plt.cla()
    x = np.linspace(0, len(mainacc), len(mainacc))
    pl.plot(x, mainacc, label="test-acc", linewidth=1.5)
    #pl.plot(x, raw_task_acc, label="target_task", linewidth=1.5)
    #pl.plot(x, trainacc, label="train-acc", linewidth=1.5)
    plt.plot(x, backdoor, label="backdoor", linewidth=1.5)
    pl.xlabel("epoch")
    pl.ylabel("acc")
    pl.legend()
    pl.savefig(datadir +'/'+ dataset+'-attack-acc.jpg')

def drawtrainacc(datadir, trainacc, mainacc, dataset='defalt'):
    # main_task_acc, raw_task_acc, trainacc, backdoor_task_acc
    '''main_task_acc
    backdoor_acc
    raw_task_acc'''
    plt.cla()
    x = np.linspace(0, len(mainacc), len(mainacc))
    pl.plot(x, mainacc, label="test-acc", linewidth=1.5)
    pl.plot(x, trainacc, label="train-acc", linewidth=1.5)
    pl.xlabel("epoch")
    pl.ylabel("acc")
    pl.legend()
    pl.savefig(datadir +'/'+ dataset+'-testtrain-acc.jpg')
def drawsuccrate(datadir, succ, succmean, dataset='defalt'):
    # main_task_acc, raw_task_acc, trainacc, backdoor_task_acc
    '''main_task_acc
    backdoor_acc
    raw_task_acc'''
    plt.cla()
    x = np.linspace(0, len(succ), len(succ))
    pl.plot(x, succ, label="successrate", linewidth=1.5)
    pl.plot(x, succmean, label="success_mean", linewidth=1.5)
    pl.xlabel("epoch")
    pl.ylabel("rate")
    pl.legend()
    pl.savefig(datadir +'/'+ dataset+'-attsuccessrate.jpg')

def drawmeanacc(datadir, mainacc, raw_task_acc, backdoor, dataset='defalt'):
    # main_task_acc, raw_task_acc, trainacc, backdoor_task_acc
    '''main_task_acc
    backdoor_acc
    raw_task_acc'''
    plt.cla()
    x = np.linspace(0, len(mainacc), len(mainacc))
    pl.plot(x, mainacc, label="test-mean", linewidth=1.5)
    pl.plot(x, raw_task_acc, label="target_mean", linewidth=1.5)
    #pl.plot(x, trainacc, label="train-acc", linewidth=1.5)
    plt.plot(x, backdoor, label="backdoor_mean", linewidth=1.5)
    pl.xlabel("epoch")
    pl.ylabel("acc")
    pl.legend()
    pl.savefig(datadir +'/'+ dataset+'-mean-acc.jpg')

def drawatack_acc(datadir, successr, dataset='defalt'):
    plt.cla()
    x = np.linspace(0, len(successr), len(successr))
    pl.plot(x, successr, label="attack-suc", linewidth=1.5)
    pl.xlabel("epoch")
    pl.ylabel("successrate")
    pl.legend()
    pl.savefig('/Users/chenshengbo/Desktop/temp/TS_FL/figure/' + datetime.now().strftime("%Y%m%d_%H%M%S") + dataset + '-attackacc.jpg')

def draw_shiyanacc(datadir, mainacc, shiyan1, shiyan2, dataset='defalt'):
    '''main_task_acc
    backdoor_acc
    raw_task_acc'''
    plt.cla()
    x = np.linspace(0, len(mainacc), len(mainacc))
    #pl.plot(x, mainacc, label="over-all-acc", linewidth=1.5)
    pl.plot(x, shiyan1, label="avg", linewidth=1.5)
    pl.plot(x, shiyan2, label="net", linewidth=1.5)
    pl.xlabel("epoch")
    pl.ylabel("acc")
    pl.legend()
    pl.savefig('/Users/chenshengbo/Desktop/temp/TS_FL/ts_attack/ts_shiyan/' +
                datetime.now().strftime("%Y%m%d_%H%M%S")+dataset+'-acc.jpg')
def four_acc(datadir, testacc2, acc, shiyantestacc, shiyantestacc2,dataset):
    plt.cla()
    x = np.linspace(0, len(testacc2), len(testacc2))
    plt.plot(x, testacc2, label="test_acc2", linewidth=1.5)
    #plt.plot(x, acc, label="train-acc", linewidth=1.5)
    plt.plot(x, shiyantestacc, label="shiyan_avg", linewidth=1.5)
    #plt.plot(x, shiyantestacc2, label="shiyan_net", linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    # plt.legend(["test_acc2", "train-acc", "shiyan_avg", "shiyan_net"])
    plt.legend()
    plt.savefig('/Users/chenshengbo/Desktop/temp/TS_FL/ts_attack/ts_shiyan/'
               + dataset + datetime.now().strftime("%Y%m%d_%H%M%S") + '-tstrainacc.jpg')

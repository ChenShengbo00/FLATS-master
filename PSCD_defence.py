import torch
from utils import *
from models.fcn import ConvNet
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
def computer_cos(vec1, vec2):
    # if np.linalg.norm(vec1) * np.linalg.norm(vec2) == 0:
    #     return 1.0
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return  cos_sim
def net_norm(net):
    norm = 0.0
    for p_index, p in enumerate(net.parameters()):
        # jisuna net list d norm
        norm += torch.norm(p.data) ** 2
    return norm
def computer_cos_model_state(model_state_list):
    juzheng = torch.ones([len(model_state_list), len(model_state_list)])
    for i in range(0, len(model_state_list)):
        for j in range(i + 1, len(model_state_list)):
            juzheng[i][j] = computer_cos(model_state_list[i], model_state_list[j])
            juzheng[j][i] = juzheng[i][j]
            #print(juzheng[j][i])
    return juzheng
def mydefence(client_models, net_freq, selected_user_indices, flr, device, maxiter=500,eps=1e-5,ftol=1e-7):
    '''计算最后一层神经网络相似度'''
    print("------------------------------------------- my defence method -------------------------------------")
    #print(len(client_models))
    fc4_list = []
    # for net_index, net in enumerate(client_models):
    #     #print('net avg net', net_norm(net))
    #     fc4_list.append(net.fc4.weight.data)
    # size = fc4_list[0].shape[0] * fc4_list[0].shape[1]
    for net_index, net in enumerate(client_models):
        temp = torch.concat((net.fc4.weight.data.flatten(), net.conv1.weight.data.flatten(), net.conv2.weight.data.flatten(), net.conv3.weight.data.flatten()), dim=0)
        #print(temp)
        fc4_list.append(temp)
    size = fc4_list[0].shape[0]
    n_class = client_models[0].n_classes
    seq_len = client_models[0].n_in

    for i in range(len(fc4_list)):
        fc4_list[i] = fc4_list[i].reshape(size).cpu()
    #print(fc4_list[0].shape)
    juzheng = torch.ones([len(fc4_list), len(fc4_list)])
    for i in range(0, len(fc4_list)):
        for j in range(i+1, len(fc4_list)):
            juzheng[i][j] = computer_cos(fc4_list[i], fc4_list[j])
            juzheng[j][i] = juzheng[i][j]

    #### 3轮模型更新之间的差异
    model_state_list = []
    for idx, global_user_idx in enumerate(selected_user_indices):
        #model_fcn = copy.deepcopy(client_models[idx])
        model_fcn = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
        model1_path = "./client_model/0" + str((flr+2) % 4) + "/" + str(global_user_idx) + "_round_model.pth"
        model_fcn.load_state_dict(torch.load(model1_path))
        ### 计算model_fcn和client_models[idx]之间的差距
        # 存储在model_fcn中
        for p_index, p in enumerate(model_fcn.parameters()):
            p.data -= list(client_models[idx].parameters())[p_index].data
        '''把模型·保存子啊来'''
        save_model_path = "./client_model/diff/" + str(global_user_idx) + "_round_model.pth"
        torch.save(model_fcn, save_model_path)
        ##### 模型参数向量化
        fcn_params = []
        for layer in model_fcn.modules():
            if isinstance(layer, nn.Conv2d):
                fcn_params.append(layer.weight.data.view(-1))
                if layer.bias is not None:
                    fcn_params.append(layer.bias.data)
            if isinstance(layer, nn.Linear):
                fcn_params.append(layer.weight.data.view(-1))
                if layer.bias is not None:
                    fcn_params.append(layer.bias.data)
        fcn_params = torch.cat(fcn_params, dim = 0).cpu()
        #print(fcn_params.shape) #torch.Size([265358])
        model_state_list.append(fcn_params)
    ####计算模型更新之间的距离
    cos_model_state = computer_cos_model_state(model_state_list)
    # print(juzheng)
    # print(cos_model_state)
    xiangsidu = 0.5 * juzheng + 0.5 * cos_model_state
    #print(xiangsidu)
    '''层次聚类，或者最近领聚类'''
    # Z = linkage(xiangsidu, method="single")
    # print(Z)
    # desired_clu = 2
    # threshold = Z[-(desired_clu-1), 2]
    # clusters = fcluster(Z, threshold, criterion="distance")
    # print(clusters)
    # for i in range(clusters.max()):
    #     clusters_index = np.where(clusters == (i+1))[0]
    #     print(clusters_index)

    distances = 1-xiangsidu
    #print(distances)
    fitmodel = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, linkage='complete')
    labels = fitmodel.fit_predict(distances)
    print("lable: ", labels)
    julei_class_is_cli = [0] * 30
    for i in range(len(labels)):
        julei_class_is_cli[labels[i]] = i
    print(julei_class_is_cli)
    class_1 = []
    class_2 = []
    for i in range(len(julei_class_is_cli)):
        if i < 0.9 * len(julei_class_is_cli):
            class_2.append(julei_class_is_cli[i])
        else :
            class_1.append(julei_class_is_cli[i])

    '''聚类法'''
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(xiangsidu)
    labels = kmeans.labels_
    class_1 = np.where(labels == 0)[0]
    class_2 = np.where(labels == 1)[0]
    if len(class_1) > len(class_2):
        temp = class_1
        class_1 = class_2
        class_2  =temp
    print("k mean labels: ", labels)
    class_num = len(set(labels))
    julei_xiangsi_list = [0.0] * class_num
    min_class = -1
    len_class = len(selected_user_indices)
    for i in range(class_num):
        indices_of_i = [index for index, label in enumerate(labels) if label == i]
        if len_class > len(indices_of_i):
            len_class = len(indices_of_i)
            min_class = i
        julei_xiangsi_list[i] = torch.mean(xiangsidu[[indices_of_i]][:,[indices_of_i]])
    #print(julei_xiangsi_list)
    min_index = julei_xiangsi_list.index(min(julei_xiangsi_list))
    min_classes = [index for index, label in enumerate(labels) if label == min_class]
    print(min_classes)
    print("xiangsidu zuixiao: ", [index for index, label in enumerate(labels) if label == min_index])
    class_1 = []
    class_2 = []
    for i in range(len(selected_user_indices)):
        if i in min_classes:
            class_1.append(i)
        else:
            class_2.append(i)
    '''相似度矩阵，剔除可疑模型'''
    # print(class_1)
    # print(class_2)
    net_list_new = []
    net_freq_new = []
    xiangsidu_chengshi = 0.0
    # 1少2多
    for i in class_2:
        xiangsidu_chengshi += xiangsidu[i].mean()
    xiangsidu_chengshi /= len(class_2)
    tichu = [False] * len(class_1)
    gama = [0] * len(class_1)
    if len(class_1) <= 0.15 * len(client_models):
        # 计算相似度差异,如果差异过大则剔除，如果差异可接受则添加客户端
        for index,i in enumerate(class_1):
            xiangsidu_i = xiangsidu[i].mean()
            gama[index] = (xiangsidu_chengshi-xiangsidu_i) / xiangsidu_chengshi
            print("gama [index] {}".format(gama[index]))
            #if abs(gama[index]) > 0.01:
            tichu[index] = True
            print("ti chu attack {}".format(selected_user_indices[i]))
            # else:
            #     class_2.append(i)

        #xiangsidu
    for i in class_2:
        if i < len(client_models):
            net_list_new.append(client_models[i])
            net_freq_new.append(net_freq[i])
    total = 0.0
    for i in  net_freq_new:
        total += i
    for i in range(len(net_freq_new)):
        net_freq_new[i] /= total

    return net_list_new, net_freq_new
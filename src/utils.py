import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import random
import torch
from sklearn.metrics import adjusted_rand_score as ari_score
import sklearn.preprocessing as preprocess
from sklearn import metrics
from scipy.io import loadmat
from model import GTN
from NMF import nmf

device = torch.device("cpu")
#------data1---------
data_path = '../dataset/'
data_set = 'data1/'

A = np.loadtxt(data_path + data_set + 'disease-lncRNA.csv',delimiter=',')
A=A.T
disSimi1 = loadmat(data_path + data_set + 'diease_similarity_kernel-k=40-t=20-ALPHA=0.1.mat') 
disSimi = disSimi1['WM']
RNASimi1 = loadmat(data_path + data_set + 'RNA_s_kernel-k=40-t=20-ALPHA=0.1.mat')
lncSimi = RNASimi1['WM']
#------data2---------
# data_path = '../dataset/'
# data_set = 'data2/'
# A = np.loadtxt(data_path + data_set + 'lncRNA-disease.txt')
# A=torch.tensor(A)
# disSimi1 = loadmat(data_path + data_set + 'diease_similarity_kernel-k=30-t=10.mat') 
# disSimi = disSimi1['WM']
# RNASimi1 = loadmat(data_path + data_set + 'RNA_s_kernel-k=40-t=10.mat')
# lncSimi = RNASimi1['WM']


def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix):
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix))
    mat2 = np.hstack((train_mirna_disease_matrix.T, disease_matrix))
    mat3= np.vstack((mat1, mat2))
    node_embeddings = torch.tensor(mat3)
    return node_embeddings

def test_features_choose(rel_adj_mat, features_embedding):
    R,D=nmf(A, lncSimi, disSimi, 0.01, 128, 500)
    RD= torch.tensor(np.vstack((R, D)))
    RD = RD.to(device)
    x = constructHNet(A, lncSimi, disSimi)
    model = GTN(7)
    b = model(x).to(device)
    b = b.squeeze().to(device)
    features_embedding = [features_embedding, RD] 
    features_embedding = torch.cat(features_embedding, dim=1)
    features_embedding = [features_embedding, b]
    features_embedding = torch.cat(features_embedding, dim=1)
    rna_nums, dis_nums = rel_adj_mat.shape[0], rel_adj_mat.shape[1]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    test_features_input, test_lable = [], []

    for i in range(rna_nums):
        for j in range(dis_nums):
            test_features_input.append((features_embedding_rna[i, :] * features_embedding_dis[j, :]).unsqueeze(0))
            test_lable.append(rel_adj_mat[i, j])

    test_features_input = torch.cat(test_features_input, dim=0)
    
    return test_features_input.to(torch.float32),test_lable

def sort_matrix(score_matrix, interact_matrix):
    '''
    实现矩阵的列元素从大到小排序
    1、np.argsort(data,axis=0)表示按列从小到大排序
    2、np.argsort(data,axis=1)表示按行从小到大排序
    '''
    sort_index = np.argsort(-score_matrix, axis=0)  # 沿着行向下(每列)的元素进行排序
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted


class Dataload(data.Dataset):

    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node
    def __getitem__(self, index):
        return index
        # adj_batch = self.Adj[index]
        # adj_mat = adj_batch[index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = self.Beta
        # return adj_batch, adj_mat, b_mat
    def __len__(self):
        return self.Node


#计算邻接矩阵
def get_adjacency_matrix(similarity_matrix, threshold):
    n = similarity_matrix.shape[0]
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] >= threshold:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix

class Sizes(object):
    def __init__(self, drug_size, mic_size):
        self.c = 12

'''def get_adjacency_matrix(feat, k):
    # compute C
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C'''
def adjacency_matrix_to_edge_index(adjacency_matrix):
  adjacency_matrix = torch.from_numpy(adjacency_matrix).clone().detach()
  num_nodes = adjacency_matrix.shape[0]
  edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t()
  return edge_index



def constructNet(mirna_disease_matrix):
    drug_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[0], mirna_disease_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[1], mirna_disease_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, mirna_disease_matrix))#沿着水平方向进行连接
    mat2 = np.hstack((mirna_disease_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))#沿着垂直方向进行连接
    return adj


def balance_samples(pos_index, neg_index):

    pos_num = len(pos_index)
    neg_num = len(neg_index)
    if pos_num > neg_num:
        # 对正样本进行下采样
        balanced_pos_index = random.sample(pos_index, neg_num)
        balanced_neg_index = neg_index
    else:
        # 对负样本进行下采样
        balanced_pos_index = pos_index
        balanced_neg_index = random.sample(neg_index, pos_num)
    return balanced_pos_index, balanced_neg_index
def random_index(index_matrix, sizes):
    set_seed(sizes.seed)
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)  # 获得相同随机数
    random.shuffle(random_index)  # 将原列表的次序打乱
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp
def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    set_seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = []
    for i in range(5):
        # 对每一折的正负样本进行平衡处理
        balanced_pos_index, balanced_neg_index = balance_samples(pos_index[i], neg_index[i])
        index.append(balanced_pos_index + balanced_neg_index)
    return index


def plot_auc_curves(fprs, tprs, aucs):
    mean_fpr = np.linspace(0, 1, 1000)
    tpr = []
    #plt.style.use('ggplot')
    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.8, label='ROC fold %d (AUC = %.4f)' % (i + 1, aucs[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    auc_std = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))
    filename = "../PT/m_f.csv"

       # 打开文件，以写入模式写入数据
    with open(filename, 'w', newline='') as csvfile:
           # 创建CSV写入器
           writer = csv.writer(csvfile)
           # 写入列表中的每个元素
           for item in mean_fpr:
               writer.writerow([item])

    filename = "../PT/m_t.csv"
       # 打开文件，以写入模式写入数据
    with open(filename, 'w', newline='') as csvfile:
           # 创建CSV写入器
           writer = csv.writer(csvfile)
           # 写入列表中的每个元素
           for item in mean_tpr:
               writer.writerow([item])
    plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle='--', color='navy', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc='lower right', prop={"size": 8})
    plt.savefig('./auc.jpg', dpi=1200, bbox_inches='tight') 
    plt.show()
    

def plot_prc_curves(precisions, recalls, auprs):
    mean_recall = np.linspace(0, 1, 1000)
    precision = []
    #plt.style.use('ggplot')
    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.8, label='ROC fold %d (AUPR = %.4f)' % (i + 1, auprs[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(auprs)
    prc_std = np.std(auprs)
    plt.plot(mean_recall, mean_precision, color='b', alpha=0.8,
             label='Mean AUPR (AUPR = %.4f $\pm$ %.4f)' % (mean_prc, prc_std))  # AP: Average Precision
    filename = "../PT/m_r.csv"
    # 打开文件，以写入模式写入数据
    with open(filename, 'w', newline='') as csvfile:
        # 创建CSV写入器
        writer = csv.writer(csvfile)
        # 写入列表中的每个元素
        for item in mean_recall:
            writer.writerow([item])

    filename = "../PT/m_p.csv"
    # 打开文件，以写入模式写入数据
    with open(filename, 'w', newline='') as csvfile:
        # 创建CSV写入器
        writer = csv.writer(csvfile)
        # 写入列表中的每个元素
        for item in mean_precision:
            writer.writerow([item])
    plt.plot([-0.05, 1.05], [1.05, -0.05], linestyle='--', color='navy', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc='lower left', prop={"size": 8})
    plt.savefig('./pr.jpg', dpi=1200, bbox_inches='tight') 
    plt.show()






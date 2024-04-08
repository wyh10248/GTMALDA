import numpy as np
import torch
from GTM_net import *
from clac_metric import get_metrics
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import gc
from scipy.io import loadmat
from utils import *
from GCNEncoder import GCNEncoder
from torch_geometric.nn import VGAE
from Loss import S_loss
from utils import  *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float,help='learning rate')
parser.add_argument('--batch_size',default=32,type=int,help="train/test batch size")
parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--k_fold', type=int, default=5, help='crossval_number.')
parser.add_argument('--epoch', type=int, default=130, help='train_number.')
parser.add_argument('--in_dim', type=int, default=256, help='in_feature.')
parser.add_argument('--out_dim', type=int, default=128, help='out_feature.')
parser.add_argument('--fout_dim', type=int, default=64, help='f-out_feature.')
parser.add_argument('--output_t', type=int, default=64, help='finally_out_feature.')
parser.add_argument('--head_num', type=int, default=8, help='head_number.')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout.')
parser.add_argument('--pos_enc_dim', type=int, default=64, help='pos_enc_dim.')
parser.add_argument('--residual', type=bool, default=True, help='RESIDUAL.')
parser.add_argument('--layer_norm', type=bool, default=True, help='LAYER_NORM.')
parser.add_argument('--batch_norm', type=bool, default=False, help='batch_norm.')
parser.add_argument('--Sa', type=int, default=8, help='TransformerLayer.') 
args = parser.parse_args()

device = torch.device("cpu")

def train(model, train_matrix, x, edge_index, optimizer, args):
    model.train()
    criterion = torch.nn.BCELoss(reduction='sum')
    def train_epoch():
        optimizer.zero_grad()  
        score =  model(x,edge_index, train_matrix)
        score = score.reshape(447,218)
        #score = score.reshape(285,226)
        loss = (criterion(score, train_matrix))
        loss.backward() 
        optimizer.step()  
        return loss
#+S_loss(x, edge_index, args)
    for epoch in range(1, args.epoch + 1):
        train_reg_loss = train_epoch()

        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss))
    pass


def PredictScore(train_matrix, x, edge_index, args):#features652*412
       train_matrix = train_matrix.contiguous()
       train_matrix = train_matrix.to(device)
       x = x.to(device)
       edge_index = edge_index.to(device)
       model = GTM_net(args, x).to(device)
       optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
       train(model, train_matrix, x, edge_index, optimizer, args)
    
       return model(x, edge_index, train_matrix)

##positive : negative = 1 : 1
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
    set_seed(args.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = []
    for i in range(sizes.k_fold):
        # 对每一折的正负样本进行平衡处理
        balanced_pos_index, balanced_neg_index = balance_samples(pos_index[i], neg_index[i])
        index.append(balanced_pos_index + balanced_neg_index)
    return index

def cross_validation_experiment(A, lncSimi, disSimi, args):
    index = crossval_index(A, args)

    metric = np.zeros((1, 7))
    score =[]
    tprs=[]
    fprs=[]
    aucs=[]
    precisions=[]
    recalls = []
    auprs = []
    pre_matrix = np.zeros(A.shape)
    print("seed=%d, evaluating lncRNA-disease...." % (args.seed))
    for k in range(args.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        
        train_matrix = np.matrix(A, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0  # 将5折中的一折变为0
        x=constructHNet(train_matrix, lncSimi, disSimi)
        edge_index=adjacency_matrix_to_edge_index(train_matrix)
        
    
        drug_len = A.shape[0]
        dis_len = A.shape[1]
        
        train_matrix = torch.from_numpy(train_matrix).to(torch.float32)
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.int64)

        drug_mic_res = PredictScore(train_matrix, x, edge_index, args)  # 预测得到的关联矩阵
        drug_mic_res= drug_mic_res.detach().cpu()
        predict_y_proba = drug_mic_res.reshape(drug_len, dis_len).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]  #从预测分数矩阵中取出验证集的预测结果 只返回相应的预测分数
        A = np.array(A)
        metric_tmp = get_metrics(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)]) #预测结果所得的评价指标
        fpr, tpr, t = roc_curve(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)])
        precision, recall, _ = precision_recall_curve(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)])
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        auprs.append(metric_tmp[1])
        print(metric_tmp)
        metric += metric_tmp      #  五折交叉验证的结果求和
        score.append(pre_matrix)
        del train_matrix  # del只删除变量，不删除数据
        gc.collect()  # 垃圾回收
    print('Mean:', metric / args.k_fold)
    metric = np.array(metric / args.k_fold)   #  五折交叉验证的结果求均值
    return metric, score, drug_len, dis_len, tprs, fprs, aucs, precisions, recalls, auprs

def main(args):
    set_seed(args.seed)
    results = []
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
    # #A=A.T
    # disSimi1 = loadmat(data_path + data_set + 'diease_similarity_kernel-k=30-t=10.mat') 
    # disSimi = disSimi1['WM']
    # RNASimi1 = loadmat(data_path + data_set + 'RNA_s_kernel-k=40-t=10.mat')
    # lncSimi = RNASimi1['WM']
    
#-------------------------
    result, score, drug_len, dis_len, tprs, fprs, aucs, precisions, recalls, auprs = cross_validation_experiment(A, lncSimi, disSimi, args)

    sizes = Sizes(drug_len, dis_len)
    score_matrix = np.mean(score, axis=0)
    np.savetxt('score.csv', score_matrix, delimiter=',')
    print(list(sizes.__dict__.values()) + result.tolist()[0][:2])
    plot_auc_curves(fprs, tprs, aucs)
    plot_prc_curves(precisions, recalls, auprs)


if __name__== '__main__':
    main(args)

 
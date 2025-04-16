import numpy as np
from utils1 import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import warnings
import pickle
from scipy.io import loadmat
import argparse
import random
import math
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

warnings.filterwarnings("ignore")
device = torch.device("cpu")

# data_path = '../case study/'
# data_set = 'data/'

# A = np.loadtxt(data_path + data_set + 'disease-lncRNA.csv',delimiter=',')
# adj_np=A.T
# num_p, num_d = adj_np.shape
# disSimi1 = loadmat(data_path + data_set + 'diease_similarity_kernel-k=40-t=20-ALPHA=0.1.mat') 
# disSimi = disSimi1['WM']
# RNASimi1 = loadmat(data_path + data_set + 'RNA_s_kernel-k=40-t=20-ALPHA=0.1.mat')
# lncSimi = RNASimi1['WM']


#   # 寻找正样本的索引
# positive_index_tuple = np.where(adj_np == 1)
# positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))

# # 随机打乱
# random.shuffle(positive_index_list)
# #将正样本分为5个数量相等的部分
# positive_split = math.ceil(len(positive_index_list) / 5)
# count = 0
# for i in range(0, len(positive_index_list), positive_split):
#         count = count + 1
#         print("This is {} fold cross validation".format(count))
#         positive_train_index_to_zero = positive_index_list[i: i + positive_split]
#         new_lncrna_disease_matrix = adj_np.copy()
#         # 五分之一的正样本置为0
#         for index in positive_train_index_to_zero:
#             new_lncrna_disease_matrix[index[0], index[1]] = 0
#         new_lncrna_disease_matrix_tensor = torch.from_numpy(new_lncrna_disease_matrix).to(torch.float32)
#         x=torch.tensor(constructHNet(new_lncrna_disease_matrix, lncSimi, disSimi)).to(torch.float32) 
#         adj=torch.tensor(constructNet(new_lncrna_disease_matrix))
#         edge_index=adjacency_matrix_to_edge_index(adj)
#         model = GTM_net(args).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
#         criterion = torch.nn.BCELoss(reduction='sum')
       
#         # 模型训练
#         model.train()
#         for epoch in range(args.epoch):
#             train_predict_result = model(x,edge_index, new_lncrna_disease_matrix_tensor)
#             pred = train_predict_result.reshape(447,218)
#             loss = (criterion(pred, new_lncrna_disease_matrix_tensor))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print('Epoch %d | train Loss: %.4f' % (epoch + 1, loss.item()))
# torch.save(model.state_dict(), 'model.pth')
# print("success")
       

#Colorectal Neoplasms,Hepatocellular Carcinoma案例研究
model = GTM_net(args).to(device)

#加载状态字典 
model.load_state_dict(torch.load('model.pth'))

#使模型处于评估模式
model.eval()

df = pd.read_csv('../case study/data/A.csv')
columns = df.columns[1:] 
print(columns.shape)
col_idx = columns.get_loc(' Hepatocellular Carcinoma')
#col_idx = columns.get_loc('colorectal Neoplasms')
#df['Colorectal Neoplasms'] = df['Colorectal Neoplasms'].replace(1, 0)
df[' Hepatocellular Carcinoma'] = df[' Hepatocellular Carcinoma'].replace(1, 0)
#去掉第一列（微生物名称），并去掉第一行（疾病名称）
adjacency_matrix = torch.from_numpy(df.iloc[:, 1:].values).to(torch.float32)
x = constructHNet(adjacency_matrix, lncSimi,disSimi).float() 
edge_index = adjacency_matrix_to_edge_index(adjacency_matrix) 
with torch.no_grad():
    pred_scores = model(x, edge_index, adjacency_matrix)
    pred = pred_scores.reshape(447,218)
    red = pred.detach().cpu().numpy()  # 转换为 NumPy 数组
    pred_df = pd.DataFrame(pred)  # 转换为 DataFrame
    pred_df.to_csv('pred.csv', index=False)  # 保存为 CSV 文件




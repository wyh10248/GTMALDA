import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import warnings
import pickle
from scipy.io import loadmat

warnings.filterwarnings("ignore")
seed_everything(42)
device = torch.device("cpu")

data_path = '../dataset/'
data_set = 'data1/'

A = np.loadtxt(data_path + data_set + 'disease-lncRNA.csv',delimiter=',')
adj_np=torch.from_numpy(A.T).to(torch.float32)
num_p, num_d = adj_np.shape
disSimi1 = loadmat(data_path + data_set + 'diease_similarity_kernel-k=40-t=20-ALPHA=0.1.mat') 
disSimi = disSimi1['WM']
RNASimi1 = loadmat(data_path + data_set + 'RNA_s_kernel-k=40-t=20-ALPHA=0.1.mat')
lncSimi = RNASimi1['WM']

x=constructHNet(adj_np, lncSimi, disSimi).to(torch.float32)
edge_index=adjacency_matrix_to_edge_index(A.T).to(torch.float32)

lr = 0.001
weight_decay = 0.0
num_epochs = 130



def fit(model,x,edge_index, A, lncSimi, disSimi, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
    criterion = torch.nn.BCELoss(reduction='sum')
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        score = model(x,edge_index, A)
        pred = score.reshape(447,218)
        loss = (criterion(pred, A))
        loss.backward()
        optimizer.step()


    model.eval()
    score = model(x,edge_index, A)
    pred = score.reshape(447,218)
    return pred

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


model = GTM_net(args, x).to(device)
model = model.to(torch.float32)
pred1 = fit(model, x, edge_index, adj_np, lncSimi, disSimi, args)
pred_np = pred1.cpu().detach().numpy()
pred = pred_np.reshape(447,218)
max_allocated_memory = torch.cuda.max_memory_allocated()
np.savetxt('pred_np_raw.csv', pred, delimiter=",")


# -*- coding: utf-8 -*-
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.io import loadmat

class GTN(nn.Module):

	def __init__(self, num_layers):
		super(GTN, self).__init__() #channels=4, layers=2
		self.num_layers = num_layers

		layers = []
		for i in range(num_layers):
			if i == 0:
				layers.append(GTLayer(first=True))
			else:
				layers.append(GTLayer(first=False))
		self.layers = nn.ModuleList(layers)
		
	def normalization(self, H):
		for i in range(1):
			if i==0:
				H_ = self.norm(H[i]).unsqueeze(0)
			else:
				H_ = torch.cat((H_,self.norm(H[i]).unsqueeze(0)), dim=0)
		return H_

	def norm(self, H, add=False):

		H = H + (torch.eye(H.shape[0]))
		deg = torch.sum(H, dim=1)
		deg[deg<=1e-10]=1
		deg_inv = deg.pow(-1)
		deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
		H = torch.mm(deg_inv,H)
		return H
	
	def forward(self, A):
		A = A.unsqueeze(0)
# auto-metapath
		for i in range(self.num_layers):#channels=4, layers=2
			if i == 0:
				H, W = self.layers[i](A)
			else:
				H = self.normalization(H)
				H, W = self.layers[i](A, H)

		return(H)

class GTLayer(nn.Module):
	
	def __init__(self, first=True):
		super(GTLayer, self).__init__()
		self.first = first
		if self.first == True:
			self.conv1 = GTConv()
			self.conv2 = GTConv()
		else:
			self.conv1 = GTConv()
	
	def forward(self, A, H_=None):
		if self.first == True:
			a = self.conv1(A)
			b = self.conv2(A)
			H = torch.bmm(a,b)
			W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
		else:
			a = self.conv1(A)
			H = torch.bmm(H_,a)
			W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
		return H,W

class GTConv(nn.Module):
	
	def __init__(self):
		super(GTConv, self).__init__()
		self.weight = nn.Parameter(torch.Tensor(2,1,1,1))#1,1代表卷积核的大小是1*1的  
		self.bias = None
		self.reset_parameters()#初始化神经网络模型的参数
	def reset_parameters(self):
		nn.init.normal_(self.weight)#对参数 weight 进行正态分布初始化。
		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, A):
		A = torch.sum(A*(F.softmax(self.weight, dim=1)), dim=0)
		return A
def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix):
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix))
    mat2 = np.hstack((train_mirna_disease_matrix.T, disease_matrix))
    mat3= np.vstack((mat1, mat2))
    node_embeddings = torch.tensor(mat3)
    return node_embeddings
# if__name__="main"
# data_path = '../dataset/' 
# data_set = 'data-R-D447-218/'
# A = np.loadtxt(data_path + data_set + 'disease-lncRNA.csv', delimiter=',')
# A=torch.tensor(A.T)
# disSimi1 = loadmat(data_path + data_set + 'diease_similarity_kernel-k=20-t=20-ALPHA=0.1.mat') 
# disSimi = disSimi1['WM']
# RNASimi1 = loadmat(data_path + data_set + 'RNA_s_kernel-k=20-t=20-ALPHA=0.1.mat')
# lncSimi = RNASimi1['WM']
# x=constructHNet(A, lncSimi, disSimi)

# model=GTN(2)
# b = model(x)
# print(b.shape)























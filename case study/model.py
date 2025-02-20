import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import argparse
from scipy.io import loadmat

device = torch.device("cpu")

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
    
def l21_norm(Mat):
    rows, cols = Mat.shape
    diagList = np.zeros(rows)

    for i in range(rows):
        add = np.sum(Mat[i, :]**2)
        diagList[i] = 1 / (2 * np.sqrt(add))

    diagTemp = np.diag(diagList)
    return diagTemp


def nmf(A, lncSimi, disSimi, beita, k, iterate):
    rows, cols = A.shape
    C = np.abs(np.random.rand(rows, k))
    D = np.abs(np.random.rand(cols, k))

    diag_cf = np.diag(np.sum(lncSimi, axis=1))
    diag_df = np.diag(np.sum(disSimi, axis=1))

    A1 = np.dot(lncSimi, A)
    A2 = np.dot(A, disSimi)
    A = np.maximum(A1, A2)

    AA = np.zeros_like(A)
    for j in range(cols):
        colList = A[:, j]
        AA[:, j] = (A[:, j] - np.min(colList)) / (np.max(colList) - np.min(colList))
    A = AA

    for step in range(iterate):
        Y = A - np.dot(C, D.T)
        B = l21_norm(Y)

        if beita > 0:
            BAD = np.dot(B, A) @ D + beita * lncSimi @ C
            BCDD = np.dot(B, C @ D.T @ D) + beita * diag_cf @ C
            C = np.multiply(C, BAD / BCDD)

        if beita > 0:
            ABC = A.T @ B @ C + beita * disSimi @ D
            DCBC = np.dot(D, C.T @ B @ C) + beita * diag_df @ D
            D = np.multiply(D, ABC / DCBC)

        scoreMat_NMF = A - np.dot(C, D.T)
        error = np.mean(np.abs(scoreMat_NMF)) / np.mean(A)
        #print(f'step={step}  error={error}')
    return C, D
def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix):
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix))
    mat2 = np.hstack((train_mirna_disease_matrix.T, disease_matrix))
    mat3= np.vstack((mat1, mat2))
    node_embeddings = torch.tensor(mat3)
    return node_embeddings
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

class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):#256,128,8
        super(MultiheadAttention, self).__init__()
        assert in_dim % num_heads == 0
        self.in_dim = in_dim
        self.hidden_dim = out_dim
        self.num_heads = num_heads
        self.depth = in_dim // num_heads#128
        self.out_dim = out_dim
        self.query_linear = nn.Linear(in_dim, in_dim)
        self.key_linear = nn.Linear(in_dim, in_dim)
        self.value_linear = nn.Linear(in_dim, in_dim)
        self.output_linear = nn.Linear(in_dim, out_dim)


    def split_heads(self, x, batch_size):
        # reshape input to [batch_size, num_heads, seq_len, depth]

        x_szie = x.size()[:-1] + (self.num_heads, self.depth)#511*8*32
        x = x.reshape(x_szie)
        return x.transpose(-1, -2)#将最后两个维度交换511*32*8

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)#652

        # Linear projections
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Split the inputs into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))#所以scores的shape最终是(1024, 128, 128),它对Q和K的乘法做了规范化处理,使得点积在0-1范围内。

        # Apply mask (if necessary)
        if mask is not None:
            mask = mask.unsqueeze(1)  # add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=0)
        attention_output = torch.matmul(attention_weights, V)

        # Merge the heads
        output_size = attention_output.size()[:-2]+ (query.size(1),)
        attention_output = attention_output.transpose(-1, -2).reshape((output_size))

        # Linear projection to get the final output
        attention_output = self.output_linear(attention_output)#1024*256

        return torch.sigmoid(attention_output)


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """
 
    def __init__(self, in_dim, hidden_dim, fout_dim, num_heads, dropout, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):#256,128,64,8,0.4,true,false,true,9
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.fout_dim = fout_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiheadAttention(in_dim, hidden_dim, num_heads)#256,128,8

        self.residual_layer1 = nn.Linear(in_dim, fout_dim)  #残差256,64

        self.O = nn.Linear(hidden_dim, fout_dim)#128*64

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(fout_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(fout_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(fout_dim, fout_dim * 2)
        self.FFN_layer2 = nn.Linear(fout_dim * 2, fout_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(fout_dim)#64

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(fout_dim)

    def forward(self, h):
        h_in1 = self.residual_layer1(h)  # for first residual connection
        # multi-head attention out
        attn_out = self.attention(h, h, h)#h=652*652,attn_out=#1024*256
        #h = attn_out.view(-1, self.out_channels)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = F.leaky_relu(self.O(attn_out))#128*64

        if self.residual:
            attn_out = h_in1 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm1(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm1(attn_out)

        h_in2 = attn_out  # for second residual connection

        # FFN
        attn_out = self.FFN_layer1(attn_out)
        attn_out = F.leaky_relu(attn_out)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = self.FFN_layer2(attn_out)
        attn_out = F.leaky_relu(attn_out)

        if self.residual:
            attn_out = h_in2 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm2(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm2(attn_out)

        return attn_out

class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result




class GTM_net(nn.Module):
    def __init__(self, args, X):
        super().__init__()
        self.X = X
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        in_dim = args.in_dim#1024
        out_dim = args.out_dim#256
        fout_dim = args.fout_dim#128
        head_num = args.head_num#8
        dropout = args.dropout#0.4 
        self.Sa = args.Sa#10
        self.output = args.output_t#64
        self.layer_norm = args.layer_norm#True
        self.batch_norm = args.batch_norm#False
        self.residual = args.residual#True
        self.FNN = nn.Linear(665, 256) 

        self.layers = nn.ModuleList([GraphTransformerLayer(in_dim, out_dim, fout_dim, head_num,dropout,
                                                            self.layer_norm, self.batch_norm, self.residual, args) for _ in range(self.Sa- 1)])#256,128,64,8,0.4,true,false,true,9
        self.layers.append(
            GraphTransformerLayer(in_dim, out_dim, fout_dim, head_num, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.FN = nn.Linear(fout_dim, self.output)
        self.mlp_prediction = MLP(857, 0.2) 
        #self.mlp_prediction = MLP(64, 0.2)
    def forward(self, x, edge_index, rel_matrix):
        x1 = self.FNN(x)
        X = F.leaky_relu(x1)
        for conv in self.layers:
            h = conv(X)
        outputs = F.leaky_relu(self.FN(h))
        
        test_features_inputs,test_lable= test_features_choose(rel_matrix, outputs)
        test_mlp_result = self.mlp_prediction(test_features_inputs)
        return test_mlp_result
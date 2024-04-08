import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE
import numpy as np
from utils import *

device = torch.device("cpu")


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
        #X = F.leaky_relu(self.hidden(x1))
        for conv in self.layers:
            h = conv(X)
        outputs = F.leaky_relu((self.FN(h)))#652*64

        test_features_inputs,test_lable= test_features_choose(rel_matrix, outputs)
        test_mlp_result = self.mlp_prediction(test_features_inputs)
        return test_mlp_result
       




import torch
from torch_geometric.nn import VGAE
from GCNEncoder import GCNEncoder
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def S_loss(x, edge_index, args):
    set_seed(args.seed)
    model_v = VGAE(GCNEncoder(665, 256)).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    z = model_v.encode(x, edge_index)
    loss = model_v.recon_loss(z, edge_index)
    Loss = loss +( 1/665)* model_v.kl_loss()

    return Loss
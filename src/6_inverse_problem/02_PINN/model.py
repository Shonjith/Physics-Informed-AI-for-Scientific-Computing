import torch
import torch.nn as nn

def make_mlp(in_dim,out_dim,hidden=128,layers=6,activation=nn.Tanh):

    seq = [nn.Linear(in_dim, hidden),activation()]

    for _ in range(layers - 2):

        seq += [nn.Linear(hidden, hidden),activation()]

    seq.append(nn.Linear(hidden, out_dim))

    return nn.Sequential(*seq)

class DualPINN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net_u = make_mlp(2,1,hidden=128,layers=6)

        self.net_S = make_mlp(1,1,hidden=64,layers=4)

    def forward_u(self,x_n,t_n):

        return self.net_u(torch.cat([x_n, t_n],dim=1))

    def forward_S(self,x_n):

        return torch.nn.functional.softplus(self.net_S(x_n))
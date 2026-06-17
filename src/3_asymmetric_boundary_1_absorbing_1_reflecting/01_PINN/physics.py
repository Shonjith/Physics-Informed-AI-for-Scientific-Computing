import torch
from data import *

def pde_residual(model, x, tau):
    x.requires_grad_(True)
    tau.requires_grad_(True)

    u     = model(x, tau)

    u_tau = torch.autograd.grad(u, tau, torch.ones_like(u), create_graph=True)[0]

    u_x = torch.autograd.grad( u, x, torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad( u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return u_tau - (D * T_max) * u_xx


def neumann_residual(model, x_b, tau_b):

    x_b.requires_grad_(True)

    u = model(x_b, tau_b)

    u_x = torch.autograd.grad( u, x_b, torch.ones_like(u), create_graph=True)[0]

    return u_x
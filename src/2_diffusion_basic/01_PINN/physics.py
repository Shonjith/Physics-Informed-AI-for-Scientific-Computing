
import torch
from data import *

def pde_loss(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(
        u, t,
        torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x,
        torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = u_t - D * u_xx
    return torch.mean(residual**2)

def ic_loss(model, x_ic):
    t0 = torch.zeros_like(x_ic).to(device)
    u_pred = model(x_ic, t0)

    u_true = A * torch.exp(-x_ic**2 / (2*sigma**2))
    return torch.mean((u_pred - u_true)**2)

def bc_loss(model, t_bc):
    x_left = (-L * torch.ones_like(t_bc)).to(device)
    x_right = (L * torch.ones_like(t_bc)).to(device)

    x_left.requires_grad_(True)
    x_right.requires_grad_(True)

    u_left = model(x_left, t_bc)
    u_right = model(x_right, t_bc)

    u_x_left = torch.autograd.grad(
        u_left, x_left,
        torch.ones_like(u_left),
        create_graph=True
    )[0]

    u_x_right = torch.autograd.grad(
        u_right, x_right,
        torch.ones_like(u_right),
        create_graph=True
    )[0]

    return torch.mean(u_x_left**2) + torch.mean(u_x_right**2)


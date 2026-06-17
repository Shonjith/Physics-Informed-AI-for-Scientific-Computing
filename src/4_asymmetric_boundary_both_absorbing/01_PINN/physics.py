import torch
import torch.nn as nn

from data import D, T_max

mse = nn.MSELoss()


# ─────────────────────────────────────────────
# PDE RESIDUAL
# ─────────────────────────────────────────────
def pde_residual(model, x, tau):

    x.requires_grad_(True)
    tau.requires_grad_(True)

    u     = model(x, tau)
    u_tau = torch.autograd.grad(u, tau, torch.ones_like(u), create_graph=True)[0]
    u_x   = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx  = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return u_tau - (D * T_max) * u_xx


# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────
def loss_function(model, data):

    x_f, t_f, x_bl, x_br, t_b, x_i, t_i, u_i = data

    # PDE residual
    f      = pde_residual(model, x_f, t_f)
    loss_f = mse(f, torch.zeros_like(f))

    # LEFT boundary: absorbing
    u_bl    = model(x_bl, t_b)
    loss_bl = mse(u_bl, torch.zeros_like(u_bl))

    # RIGHT boundary: absorbing
    u_br    = model(x_br, t_b)
    loss_br = mse(u_br, torch.zeros_like(u_br))

    # Initial condition
    u_pred_i = model(x_i, t_i)
    loss_i   = mse(u_pred_i, u_i)

    return loss_f + 100 * loss_bl + 100 * loss_br + 500 * loss_i
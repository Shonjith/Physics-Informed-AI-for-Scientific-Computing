import torch
import torch.nn as nn

from data import ( D, T_max, device)

mse = nn.MSELoss()

# ─────────────────────────────────────────────
# PDE RESIDUAL
# ─────────────────────────────────────────────
def pde_residual_2d(model, x, y, tau):

    x   = x.clone().requires_grad_(True)
    y   = y.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)

    u = model(x, y, tau)

    u_tau = torch.autograd.grad(u, tau,torch.ones_like(u),create_graph=True)[0]

    u_x = torch.autograd.grad(u, x,torch.ones_like(u),create_graph=True)[0]

    u_y = torch.autograd.grad(u, y,torch.ones_like(u),create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x,torch.ones_like(u_x),create_graph=True)[0]

    u_yy = torch.autograd.grad(u_y, y,torch.ones_like(u_y),create_graph=True)[0]

    return u_tau - D * T_max * (u_xx + u_yy)


# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────
def loss_function_2d(model,data,w_pde=1.0,w_bc=50.0,w_ic=200.0):

    (
        x_f, y_f, tau_f,x_bl, y_bl,x_br, y_br,
        x_bb, y_bb,x_bt, y_bt,tau_b,x_i, y_i,tau_i, u_i
    ) = data

    # PDE loss
    res = pde_residual_2d(model,x_f,y_f,tau_f)

    loss_f = mse(
        res,
        torch.zeros_like(res)
    )

    # Boundary loss
    def bc_loss(xb, yb):

        return mse(
            model(xb, yb, tau_b),
            torch.zeros(
                xb.shape[0],
                1,
                device=device
            )
        )

    loss_b = (bc_loss(x_bl, y_bl)+ bc_loss(x_br, y_br)+ bc_loss(x_bb, y_bb)+ bc_loss(x_bt, y_bt))

    # Initial condition loss
    u_pred = model(x_i,y_i,tau_i)

    loss_i = mse(u_pred, u_i)

    total = (w_pde * loss_f+ w_bc * loss_b+ w_ic * loss_i)

    return (total,loss_f.item(),loss_b.item(),loss_i.item())
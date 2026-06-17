import torch

from data import (x_min,x_max,t_min,t_max)

def physics_loss(model,xc_n,tc_n,D):

    xc_n = xc_n.requires_grad_(True)

    tc_n = tc_n.requires_grad_(True)

    u = model.forward_u(xc_n,tc_n)

    S = model.forward_S(xc_n.detach().requires_grad_(False))

    grads_u = torch.autograd.grad(u,[xc_n, tc_n],grad_outputs=torch.ones_like(u),create_graph=True)[:]

    u_x = grads_u[0]

    u_t = grads_u[1]

    u_xx = torch.autograd.grad(u_x,xc_n,grad_outputs=torch.ones_like(u_x),create_graph=True)[0]

    u_t_phys = (u_t / (t_max - t_min))

    u_xx_phys = (u_xx / (x_max - x_min)**2)

    S_at_col = model.forward_S(xc_n.detach())

    residual = (u_t_phys - D * u_xx_phys + S_at_col * u)

    return (residual**2).mean()
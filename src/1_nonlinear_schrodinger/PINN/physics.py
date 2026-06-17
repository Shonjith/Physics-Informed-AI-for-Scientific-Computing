import torch

def gradients(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

def schrodinger_residual(model, t, x):
    tx = torch.cat([t, x], dim=1)
    uv = model(tx)

    u = uv[:, 0:1]
    v = uv[:, 1:2]

    u_t = gradients(u, t)
    v_t = gradients(v, t)

    u_x = gradients(u, x)
    v_x = gradients(v, x)

    u_xx = gradients(u_x, x)
    v_xx = gradients(v_x, x)

    f_u = -v_t + 0.5 * u_xx + (u**2 + v**2) * u
    f_v =  u_t + 0.5 * v_xx + (u**2 + v**2) * v

    return f_u, f_v

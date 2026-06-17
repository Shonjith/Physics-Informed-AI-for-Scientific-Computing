import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L     = 5.0
D     = 0.1
T_max = 900.0

x_left        = -L
x_right       = 5 * L
domain_length = x_right - x_left

x0    = 0.0
sigma = 1.0


def initial_condition(x):
    u    = torch.exp(-(x - x0) ** 2 / sigma ** 2)
    norm = sigma * math.sqrt(math.pi)
    return u / norm


def sample_points(N_f=5000, N_b=1000, N_i=1000):

    x_f   = x_left + torch.rand(N_f, 1) * domain_length
    tau_f = torch.rand(N_f, 1)

    tau_b       = torch.rand(N_b, 1)
    x_b_left    = x_left  * torch.ones(N_b, 1)
    x_b_right   = x_right * torch.ones(N_b, 1)

    x_i_uniform = x_left + torch.rand(N_i // 2, 1) * domain_length
    x_i_focused = torch.randn(N_i // 2, 1) * sigma + x0

    x_i = torch.cat([x_i_uniform, x_i_focused], dim=0)
    x_i = torch.clamp(x_i, x_left, x_right)

    tau_i = torch.zeros(N_i, 1)
    u_i   = initial_condition(x_i)

    return (
        x_f.to(device), tau_f.to(device),
        x_b_left.to(device), x_b_right.to(device), tau_b.to(device),
        x_i.to(device), tau_i.to(device), u_i.to(device),
    )
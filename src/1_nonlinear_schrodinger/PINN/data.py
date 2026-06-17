import torch
import numpy as np

def initial_points(N, device):
    x = np.random.uniform(-5, 5, (N, 1))
    t = np.zeros_like(x)

    u0 = 2 / np.cosh(x)
    v0 = np.zeros_like(x)

    return (
        torch.tensor(t, requires_grad=True).float().to(device),
        torch.tensor(x, requires_grad=True).float().to(device),
        torch.tensor(u0).float().to(device),
        torch.tensor(v0).float().to(device)
    )


def collocation_points(N, device):
    t = np.random.uniform(0, np.pi/2, (N, 1))
    x = np.random.uniform(-5, 5, (N, 1))

    return (
        torch.tensor(t, requires_grad=True).float().to(device),
        torch.tensor(x, requires_grad=True).float().to(device)
    )


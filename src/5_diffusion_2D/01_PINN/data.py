import torch
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# DOMAIN & PARAMETERS
# ─────────────────────────────────────────────
D     = 0.1
T_max = 50.0

x_lo, x_hi = -5.0, 5.0
y_lo, y_hi = -5.0, 5.0

Lx = x_hi - x_lo
Ly = y_hi - y_lo

x0, y0 = 0.0, 0.0
sigma  = 1.0


def initial_condition_2d(x, y):

    u = torch.exp(-((x - x0)**2 + (y - y0)**2)/ sigma**2)

    norm = math.pi * sigma**2

    return u / norm


# ─────────────────────────────────────────────
# COLLOCATION POINT SAMPLING
# ─────────────────────────────────────────────
def sample_points_2d(N_f=15000,N_b=2000,N_i=4000):

    # Interior
    x_f   = x_lo + torch.rand(N_f, 1) * Lx
    y_f   = y_lo + torch.rand(N_f, 1) * Ly
    tau_f = torch.sqrt(torch.rand(N_f, 1))

    # Boundary time sampling
    N_b_half = N_b // 2

    tau_b_early = torch.sqrt(torch.rand(N_b_half, 1))
    tau_b_late  = torch.rand(N_b_half, 1)

    tau_b = torch.cat([tau_b_early, tau_b_late],dim=0)

    # Boundaries
    x_bl = x_lo * torch.ones(N_b, 1)
    y_bl = y_lo + torch.rand(N_b, 1) * Ly

    x_br = x_hi * torch.ones(N_b, 1)
    y_br = y_lo + torch.rand(N_b, 1) * Ly

    x_bb = x_lo + torch.rand(N_b, 1) * Lx
    y_bb = y_lo * torch.ones(N_b, 1)

    x_bt = x_lo + torch.rand(N_b, 1) * Lx
    y_bt = y_hi * torch.ones(N_b, 1)

    # Initial condition
    third = N_i // 3
    rem   = N_i - 2 * third

    x_iu = x_lo + torch.rand(third, 1) * Lx
    y_iu = y_lo + torch.rand(third, 1) * Ly

    x_ic1 = torch.clamp(torch.randn(third, 1) * sigma + x0,x_lo,x_hi)

    y_ic1 = torch.clamp(torch.randn(third, 1) * sigma + y0,y_lo,y_hi)

    x_ic2 = torch.clamp(torch.randn(rem, 1) * 3 * sigma + x0,x_lo,x_hi)

    y_ic2 = torch.clamp(torch.randn(rem, 1) * 3 * sigma + y0,y_lo,y_hi)

    x_i = torch.cat([x_iu, x_ic1, x_ic2], dim=0)
    y_i = torch.cat([y_iu, y_ic1, y_ic2], dim=0)

    tau_i = torch.zeros(N_i, 1)

    u_i = initial_condition_2d(x_i, y_i)

    def _d(*ts):
        return tuple(t.to(device) for t in ts)

    return (
        *_d(x_f, y_f, tau_f),

        *_d( x_bl, y_bl, x_br, y_br, x_bb, y_bb, x_bt, y_bt, tau_b),

        *_d(x_i, y_i, tau_i, u_i)
    )
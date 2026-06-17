import torch
import numpy as np

from data import (device,D,T_max,x_lo,x_hi,y_lo,y_hi,Lx,Ly)

# ─────────────────────────────────────────────
# EVALUATE PINN OVER GRID
# ─────────────────────────────────────────────
def evaluate_model(model):

    Nx, Ny = 200, 200

    dx = (x_hi - x_lo) / (Nx - 1)
    dy = (y_hi - y_lo) / (Ny - 1)

    dA = dx * dy

    x_arr = np.linspace(x_lo, x_hi, Nx)
    y_arr = np.linspace(y_lo, y_hi, Ny)

    XX, YY = np.meshgrid(x_arr, y_arr, indexing='ij')

    X_flat = torch.tensor(XX.ravel(), dtype=torch.float32).view(-1, 1).to(device)
    Y_flat = torch.tensor(YY.ravel(), dtype=torch.float32).view(-1, 1).to(device)

    r2_arr = (XX**2 + YY**2).ravel()

    snap_times  = [0, 1, 5, 10, 25, 50]
    snap_fields = {}

    t_rec = np.unique(np.concatenate((np.linspace(0, T_max, 500), snap_times)))
    t_rec = np.sort(t_rec)

    S_rec   = []
    msd_rec = []

    print("Evaluating PINN over grid...")

    with torch.no_grad():

        for t in t_rec:

            tau_val  = t / T_max
            Tau_flat = torch.full_like(X_flat, tau_val)

            u_pred = model(X_flat, Y_flat, Tau_flat).cpu().numpy().ravel()

            S = np.sum(u_pred) * dA

            if S > 1e-15:
                msd = np.dot(u_pred, r2_arr) * dA / S
            else:
                msd = np.nan

            S_rec.append(S)
            msd_rec.append(msd)

            for st in snap_times:

                if np.isclose(t, st, atol=1e-3) and st not in snap_fields:
                    snap_fields[st] = u_pred.reshape(Nx, Ny)

    S_rec   = np.array(S_rec)
    msd_rec = np.array(msd_rec)

    return (x_arr,y_arr,snap_times,snap_fields,t_rec,S_rec,msd_rec,Lx,Ly,D)

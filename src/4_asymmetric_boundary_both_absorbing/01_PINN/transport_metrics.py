import torch
import numpy as np
from sklearn.linear_model import LinearRegression

from data import ( x_left, x_right, T_max, D, device)

# ─────────────────────────────────────────────
# SURVIVAL PROBABILITY & MSD COMPUTATION
# ─────────────────────────────────────────────
x_0 = 0.0


def compute_metrics(model, t_val):

    x = np.linspace(x_left, x_right, 2000)

    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)

    tau_val  = t_val / T_max
    tau_tens = torch.ones_like(x_tensor) * tau_val

    u = model(x_tensor, tau_tens).detach().cpu().numpy().flatten()

    # Survival Probability
    S_t = np.trapz(u, x)

    # Normalized MSD
    if S_t > 1e-12:

        u_norm = u / S_t

        msd_t = np.trapz((x - x_0)**2 * u_norm, x)

    else:
        msd_t = np.nan

    return S_t, msd_t


def compute_transport_metrics(model):

    t_vals = np.linspace(1, T_max, 900)

    print("Computing metrics ...")

    results = [compute_metrics(model, t) for t in t_vals]

    S_vals   = np.array([r[0] for r in results])
    MSD_vals = np.array([r[1] for r in results])

    return t_vals, S_vals, MSD_vals


# ─────────────────────────────────────────────
# TRANSPORT CLASSIFICATION: MSD
# ─────────────────────────────────────────────
def classify_transport_msd(t_vals, MSD_vals):

    fit_start_frac_msd = 0.10

    mask_msd = (
        (t_vals >= fit_start_frac_msd * T_max)
        & (~np.isnan(MSD_vals))
    )

    log_t_msd = np.log(t_vals[mask_msd]).reshape(-1, 1)
    log_msd   = np.log(MSD_vals[mask_msd])

    reg_msd = LinearRegression().fit(log_t_msd, log_msd)

    alpha = reg_msd.coef_[0]

    if alpha < 0.90:
        classification = "Subdiffusion"

    elif alpha > 1.10:
        classification = "Superdiffusion/Ballistic"

    else:
        classification = "Normal Diffusion"

    print(f"MSD Exponent α = {alpha:.4f} --> {classification}")

    return alpha, classification, reg_msd, mask_msd


# ─────────────────────────────────────────────
# LATE-TIME DECAY
# ─────────────────────────────────────────────
def compute_late_time_decay(t_vals, S_vals):

    L = x_right - x_left

    true_slope = -D * (np.pi / L)**2

    fit_start_frac_s = 0.70

    mask_s = (
        (t_vals >= fit_start_frac_s * T_max)
        & (S_vals > 1e-12)
    )

    t_s = t_vals[mask_s].reshape(-1, 1)

    log_S_s = np.log(S_vals[mask_s])

    reg_s = LinearRegression().fit(t_s, log_S_s)

    computed_slope = reg_s.coef_[0]

    print(f"Computed Late-Time Slope = {computed_slope:.6f}")
    print(f"True Slope (Theoretical) = {true_slope:.6f}")

    return computed_slope, true_slope, reg_s, mask_s
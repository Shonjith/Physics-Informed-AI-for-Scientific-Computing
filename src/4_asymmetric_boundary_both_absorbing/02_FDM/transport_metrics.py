import numpy as np
from sklearn.linear_model import LinearRegression

from solver import (T_max,D,x_left,x_right)

# ─────────────────────────────────────────────
# MSD TRANSPORT CLASSIFICATION
# ─────────────────────────────────────────────
def classify_transport_msd(t_vals, msd_vals):

    print("Computing transport classification...")

    fit_start_frac_msd = 0.10

    mask_msd = ((t_vals >= fit_start_frac_msd * T_max) & (~np.isnan(msd_vals)))

    log_t_msd = np.log(t_vals[mask_msd]).reshape(-1, 1)
    log_msd   = np.log(msd_vals[mask_msd])

    reg_msd = LinearRegression().fit(log_t_msd, log_msd)

    alpha_exp = reg_msd.coef_[0]

    # Classification
    if alpha_exp < 0.95:
        classification = "Subdiffusion"

    elif alpha_exp > 1.05:
        classification = "Superdiffusion / Ballistic"

    else:
        classification = "Normal Diffusion"

    print(f"  MSD Exponent α = {alpha_exp:.4f}")
    print(f"  Classification = {classification}\n")

    return alpha_exp, classification, reg_msd, mask_msd


# ─────────────────────────────────────────────
# LATE-TIME SURVIVAL DECAY
# ─────────────────────────────────────────────
def compute_late_time_decay(t_vals, S_vals):

    print("Computing late-time survival decay...")

    # Theoretical slope
    L_domain = x_right - x_left

    true_slope = -D * (np.pi / L_domain)**2

    fit_start_frac_s = 0.70

    mask_s = ((t_vals >= fit_start_frac_s * T_max) & (S_vals > 1e-12))

    t_s = t_vals[mask_s].reshape(-1, 1)

    log_S_s = np.log(S_vals[mask_s])

    reg_s = LinearRegression().fit(t_s, log_S_s)

    computed_slope = reg_s.coef_[0]

    print(f"  Computed Late-Time Slope = {computed_slope:.6f}")
    print(f"  True Theoretical Slope   = {true_slope:.6f}\n")

    return computed_slope, true_slope, reg_s, mask_s
import numpy as np
from sklearn.linear_model import LinearRegression

from solver import T_max

def classify_transport(t_vals, S_vals):

    fit_start_frac = 0.10

    mask  = t_vals >= fit_start_frac * T_max
    log_t = np.log(t_vals[mask]).reshape(-1, 1)

    log_S = np.log(np.abs(S_vals[mask]) + 1e-12)

    reg   = LinearRegression().fit(log_t, log_S)

    slope = reg.coef_[0]
    beta  = -slope

    print(f"\nlog-log slope = {slope:.4f}")
    print(f"Decay exponent β = {beta:.4f}")

    if abs(beta - 0.5) < 0.15:
        transport_type = "Normal Diffusion (β ≈ 0.5, α = 1)"

    elif beta < 0.5:
        transport_type = f"Subdiffusion (β ≈ {beta:.2f} < 0.5)"

    else:
        transport_type = f"Ballistic / Superdiffusion (β ≈ {beta:.2f} > 0.5)"

    print(f"Classification → {transport_type}")

    return slope, beta, reg, transport_type, mask
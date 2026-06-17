import torch
import numpy as np
from sklearn.linear_model import LinearRegression

from data import x_left, x_right, T_max, device

# ─────────────────────────────────────────────
# SURVIVAL PROBABILITY
# ─────────────────────────────────────────────
def survival_probability(model, t_val):

    x = np.linspace(x_left, x_right, 2000)

    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)

    tau_val  = t_val / T_max
    tau_tens = torch.ones_like(x_tensor) * tau_val

    u = model(x_tensor, tau_tens).detach().cpu().numpy().flatten()

    return np.trapz(u, x)


def compute_survival_curve(model):

    t_vals = np.linspace(1, T_max, 900)

    print("Computing survival probabilities ...")

    S_vals = np.array([survival_probability(model, t) for t in t_vals])

    return t_vals, S_vals


# ─────────────────────────────────────────────
# TRANSPORT CLASSIFICATION
# ─────────────────────────────────────────────
def classify_transport(t_vals, S_vals, fit_start_frac=0.10):

    mask  = t_vals >= fit_start_frac * T_max
    log_t = np.log(t_vals[mask]).reshape(-1, 1)

    log_S = np.log(np.abs(S_vals[mask]) + 1e-12)

    reg   = LinearRegression().fit(log_t, log_S)

    slope = reg.coef_[0]
    beta  = -slope

    print(f"log-log slope = {slope:.4f}")
    print(f"Decay exponent β = {beta:.4f}")

    if abs(beta - 0.5) < 0.15:
        transport_type = "Normal Diffusion (β ≈ 0.5, α = 1)"

    elif beta < 0.5:
        transport_type = f"Subdiffusion (β ≈ {beta:.2f} < 0.5)"

    else:
        transport_type = f"Ballistic / Superdiffusion (β ≈ {beta:.2f} > 0.5)"

    print(f"Classification → {transport_type}")

    return slope, beta, reg, transport_type, mask
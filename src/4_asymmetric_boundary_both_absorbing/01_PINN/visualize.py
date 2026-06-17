import torch
import numpy as np
import matplotlib.pyplot as plt

from data import ( x_left, x_right, T_max, device)

# ─────────────────────────────────────────────
# SOLUTION SNAPSHOTS
# ─────────────────────────────────────────────
def plot_solution_snapshots(model):

    x_plot   = np.linspace(x_left, x_right, 300)

    x_plot_t = torch.tensor(x_plot, dtype=torch.float32).view(-1, 1).to(device)

    plt.figure(figsize=(9, 5))

    for t_val in [10, 100, 300, 700]:

        tau_val    = t_val / T_max
        tau_tensor = torch.ones_like(x_plot_t) * tau_val

        u = model(x_plot_t, tau_tensor).detach().cpu().numpy()

        plt.plot(x_plot, u, label=f"t = {t_val}")

    plt.axvline(x=x_left, color='red',
                linestyle='--', linewidth=1.2,
                label="Absorbing (x = −5)")

    plt.axvline(x=x_right, color='red',
                linestyle='--', linewidth=1.2,
                label="Absorbing (x = 25)")

    plt.legend()

    plt.title("u(x, t) — Both boundaries absorbing")

    plt.xlabel("x")
    plt.ylabel("u")

    plt.tight_layout()

    plt.savefig("solution_snapshots_two_absorbing.png", dpi=150)

    plt.show()


# ─────────────────────────────────────────────
# SURVIVAL PROBABILITY
# ─────────────────────────────────────────────
def plot_survival_probability(t_vals, S_vals):

    plt.figure(figsize=(8, 5))

    plt.plot(t_vals, S_vals, color="steelblue", lw=2)

    plt.title("Survival Probability S(t)")
    plt.xlabel("Physical time t")
    plt.ylabel("S(t)")

    plt.tight_layout()

    plt.savefig("survival_probability.png", dpi=150)

    plt.show()


# ─────────────────────────────────────────────
# MSD CLASSIFICATION
# ─────────────────────────────────────────────
def plot_msd_classification(t_vals,MSD_vals,alpha,classification,reg_msd,mask_msd):

    plt.figure(figsize=(8, 5))

    plt.scatter(
        np.log(t_vals),
        np.log(MSD_vals),
        s=6,
        alpha=0.6,
        color="purple",
        label="log MSD(t)"
    )

    t_line_msd = np.linspace(
        np.log(t_vals[mask_msd]).min(),
        np.log(t_vals[mask_msd]).max(),
        200
    )

    plt.plot(
        t_line_msd,
        reg_msd.intercept_ + alpha * t_line_msd,
        color="red",
        lw=2,
        label=f"Fit α = {alpha:.3f}"
    )

    plt.xlabel("log t")
    plt.ylabel("log MSD(t)")

    plt.title(f"Transport Classification: {classification}")

    plt.legend()

    plt.tight_layout()

    plt.savefig("msd_classification.png", dpi=150)

    plt.show()


# ─────────────────────────────────────────────
# LATE-TIME DECAY
# ─────────────────────────────────────────────
def plot_logS_decay(t_vals,S_vals,computed_slope,true_slope,reg_s,mask_s):

    plt.figure(figsize=(8, 5))

    plt.plot(
        t_vals,
        np.log(S_vals + 1e-12),
        color="darkgreen",
        label="log S(t)"
    )

    t_line_s = np.linspace(t_vals[mask_s][0], t_vals[-1], 200)

    plt.plot(
        t_line_s,
        reg_s.intercept_ + computed_slope * t_line_s,
        color="red",
        lw=2,
        linestyle="--",
        label=f"Computed Fit = {computed_slope:.4f}"
    )

    plt.plot(
        t_line_s,
        reg_s.intercept_ + true_slope * t_line_s,
        color="blue",
        lw=2,
        linestyle=":",
        label=f"True Slope = {true_slope:.4f}"
    )

    plt.xlabel("t")
    plt.ylabel("log S(t)")

    plt.title("Late-Time Exponential Decay of Survival Probability")

    plt.legend()

    plt.tight_layout()

    plt.savefig("logS_vs_t_decay.png", dpi=150)

    plt.show()
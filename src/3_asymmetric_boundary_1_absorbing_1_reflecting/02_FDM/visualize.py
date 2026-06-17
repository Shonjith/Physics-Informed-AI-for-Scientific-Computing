import numpy as np
import matplotlib.pyplot as plt

from solver import (
    x,
    x_left,
    x_right
)

def plot_solution_snapshots(snapshots):

    plt.figure(figsize=(9, 5))

    for t_val in [10, 100, 300, 700]:

        if t_val in snapshots:
            plt.plot(x, snapshots[t_val], label=f"t = {t_val}")

    plt.axvline(x=x_left, color='red', linestyle='--',
                linewidth=1.2, label="Absorbing (x = −5)")

    plt.axvline(x=x_right, color='blue', linestyle='--',
                linewidth=1.2, label="Reflective (x = 25)")

    plt.legend()

    plt.title("Explicit FDM: u(x, t) — one absorbing, one reflective boundary")

    plt.xlabel("x")
    plt.ylabel("u")

    plt.tight_layout()

    plt.savefig("fdm_solution_snapshots.png", dpi=150)

    plt.show()

def plot_survival_probability(t_vals, S_vals):

    plt.figure(figsize=(8, 5))

    plt.plot(t_vals, S_vals, color="steelblue")

    plt.title("Explicit FDM: Survival Probability S(t)")

    plt.xlabel("Physical time t")
    plt.ylabel("S(t)")

    plt.tight_layout()

    plt.savefig("fdm_survival_probability.png", dpi=150)

    plt.show()

def plot_transport_classification(
    t_vals, S_vals, slope, beta, reg, mask, transport_type
):

    plt.figure(figsize=(8, 5))

    plt.scatter(np.log(t_vals), np.log(np.abs(S_vals) + 1e-12),
                s=6, alpha=0.6, color="steelblue", label="log S(t)")

    t_fit_line = np.linspace(np.log(t_vals[mask][0]),
                             np.log(t_vals[-1]), 200)

    plt.plot(t_fit_line, reg.intercept_ + slope * t_fit_line,
             color="red", linewidth=2,
             label=f"Fit slope = {slope:.3f} (β = {beta:.3f})")

    plt.xlabel("log t")
    plt.ylabel("log S(t)")

    plt.title(f"Explicit FDM: Transport Classification\n→ {transport_type}")

    plt.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig("fdm_transport_classification.png", dpi=150)

    plt.show()
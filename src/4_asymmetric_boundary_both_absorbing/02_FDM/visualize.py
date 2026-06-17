import numpy as np
import matplotlib.pyplot as plt

from solver import (x_left,x_right)

# ─────────────────────────────────────────────
# SNAPSHOTS
# ─────────────────────────────────────────────
def plot_solution_snapshots(x, snapshots):

    plt.figure(figsize=(9, 5))

    for t_val in [10, 100, 300, 700]:

        if t_val in snapshots:
            plt.plot(x, snapshots[t_val], label=f"t = {t_val}")

    plt.axvline(x=x_left,color='red',linestyle='--',linewidth=1.2,label=f"Absorbing (x = {x_left})")

    plt.axvline(x=x_right,color='red',linestyle='--',linewidth=1.2,label=f"Absorbing (x = {x_right})")

    plt.legend()

    plt.title("FDM Solution u(x, t) — Both boundaries absorbing")

    plt.xlabel("x")
    plt.ylabel("u")

    plt.tight_layout()

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

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


# ─────────────────────────────────────────────
# MSD CLASSIFICATION
# ─────────────────────────────────────────────
def plot_msd_classification(t_vals,msd_vals,alpha_exp,classification,reg_msd,mask_msk):

    plt.figure(figsize=(8, 5))

    plt.scatter(np.log(t_vals),np.log(msd_vals),s=6,alpha=0.6,color="purple",label="log MSD(t)")

    t_line_msd = np.linspace(np.log(t_vals[mask_msd]).min(),np.log(t_vals[mask_msd]).max(), 200)

    plt.plot(
        t_line_msd,reg_msd.intercept_ + alpha_exp * t_line_msd,color="red",lw=2,
        label=f"Fit α = {alpha_exp:.3f}"
        )

    plt.xlabel("log t")
    plt.ylabel("log MSD(t)")

    plt.title(f"Transport Classification: {classification}")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


# ─────────────────────────────────────────────
# LATE-TIME DECAY
# ─────────────────────────────────────────────
def plot_logS_decay(t_vals,S_vals,computed_slope,true_slope,reg_s,mask_s):

    plt.figure(figsize=(8, 5))

    plt.plot(t_vals,np.log(S_vals + 1e-12),color="darkgreen",label="log S(t)",lw=2)

    t_line_s = np.linspace(t_vals[mask_s][0], t_vals[-1], 200)

    plt.plot(
        t_line_s,reg_s.intercept_ + computed_slope * t_line_s,color="red",
        lw=2,linestyle="--",label=f"Computed Fit Slope = {computed_slope:.5f}"
    )

    plt.plot(
        t_line_s,reg_s.intercept_ + true_slope * t_line_s,color="blue",
        lw=2,linestyle=":",label=f"True Slope = {true_slope:.5f}"
    )

    plt.xlabel("Physical time t")
    plt.ylabel("log S(t)")

    plt.title("Late-Time Exponential Decay of Survival Probability")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()
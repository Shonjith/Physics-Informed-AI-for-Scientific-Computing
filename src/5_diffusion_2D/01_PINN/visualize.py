import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# HEATMAPS
# ─────────────────────────────────────────────
def plot_heatmaps(x_arr,y_arr,snap_times,snap_fields):

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes = axes.ravel()

    for ax, t_val in zip(axes, snap_times):

        field = snap_fields.get(t_val, None)

        if field is None:
            ax.set_title(f"t = {t_val} (missing)")
            continue

        field_clipped = np.clip(field, 0, None).T

        im = ax.contourf(x_arr,y_arr,field_clipped,levels=50,cmap="inferno")

        fig.colorbar(im, ax=ax)

        ax.set_title(f"t = {t_val}", fontsize=11)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        for spine in ax.spines.values():
            spine.set_edgecolor("cyan")
            spine.set_linewidth(1.5)

    fig.suptitle("PINN 2D Diffusion — Heatmaps", fontsize=13)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MSD CLASSIFICATION
# ─────────────────────────────────────────────
def plot_transport_classification(
    t_rec,msd_corrected,valid_t,
    mask,alpha,transport,intercept
):

    plt.figure(figsize=(8, 5))

    plt.scatter(
        np.log(t_rec[valid_t]),np.log(msd_corrected[valid_t]),s=6,
        alpha=0.6,color="blue",label="PINN log MSD_corr(t)"
    )

    if mask.sum() >= 10:

        t_line = np.linspace(np.log(t_rec[mask][0]),np.log(t_rec[mask][-1]),300)

        msd_line = intercept + (alpha * t_line)

        plt.plot(
            t_line, msd_line, color="orange",
            lw=2.5, label=f"Fit α = {alpha:.3f} ({transport})"
        )

    plt.xlabel("log t")
    plt.ylabel("log (MSD - MSD₀)")

    plt.title(
        f"PINN Transport Classification\n"
        f"→ {transport} (Early Time Regime)"
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# SURVIVAL + LATE TIME DECAY
# ─────────────────────────────────────────────
def plot_survival_decay(
    t_rec,S_rec_smooth,t_validS,log_S,
    late_mask,reg_S,sim_slope,sim_intercept,ana_slope
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Survival Probability
    ax1.plot(t_rec,S_rec_smooth,color="darkorange",lw=2)

    ax1.set_title(
        "PINN Survival Probability S(t) "
        "(Smoothed Early Regime)"
    )

    ax1.set_xlabel("Physical time  t")
    ax1.set_ylabel("S(t)")

    ax1.set_ylim(bottom=0, top=1.05)

    ax1.grid(True, linestyle='--', alpha=0.6)

    # Log Survival
    ax2.plot(t_validS,log_S,color="purple",lw=2,label="PINN ln(S(t))")


    if late_mask.sum() > 10:

        t_late = t_validS[late_mask].reshape(-1, 1)

        ax2.plot(
            t_validS[late_mask],reg_S.predict(t_late),color="cyan",
            lw=2,linestyle='--',label=f"PINN Fit Slope: {sim_slope:.5f}"
        )

        ana_line = (
            ana_slope * t_validS[late_mask]
            + (sim_intercept + (sim_slope - ana_slope) * t_validS[late_mask][0])
        )

        ax2.plot(
            t_validS[late_mask],ana_line,color="red",
            lw=2,linestyle=':',label=f"Analytical Slope: {ana_slope:.5f}"
        )

    ax2.set_title(
        "PINN Log Survival Probability "
        "vs Time (Late Regime)"
    )

    ax2.set_xlabel("Physical time  t")
    ax2.set_ylabel("ln(S(t))")

    ax2.legend()

    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
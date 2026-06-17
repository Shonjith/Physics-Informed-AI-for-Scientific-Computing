import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# HEATMAPS
# ─────────────────────────────────────────────
def plot_heatmaps(x_arr,y_arr,snap_times,snap_indices,snap_fields):

    fig, axes = plt.subplots(2,3,figsize=(14, 9))

    axes = axes.ravel()

    for ax, t_val in zip(axes, snap_times):

        idx = snap_indices[snap_times.index(t_val)]

        field = snap_fields.get(idx, None)

        if field is None:
            ax.set_title(
                f"t = {t_val} (missing)"
            )
            continue

        field_clipped = np.clip(field,0,None).T

        im = ax.contourf(x_arr,y_arr,field_clipped,levels=50,cmap="inferno")

        plt.colorbar(im, ax=ax)

        ax.set_title(f"t = {t_val}",fontsize=11)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        for spine in ax.spines.values():
            spine.set_edgecolor("cyan")
            spine.set_linewidth(1.5)

    fig.suptitle(
        "2D Diffusion — FTCS Explicit FD, "
        "4 Absorbing Boundaries",
        fontsize=13
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# TRANSPORT CLASSIFICATION
# ─────────────────────────────────────────────
def plot_transport_classification(
    t_rec,msd_corrected,valid_t,
    mask,reg,alpha,transport
):

    plt.figure(figsize=(8, 5))

    plt.scatter(
        np.log(t_rec[valid_t]),np.log(msd_corrected[valid_t]),s=6,
        alpha=0.6,color="dodgerblue",label="log MSD_corr(t)"
    )

    if mask.sum() >= 10:

        t_line = np.linspace(np.log(t_rec[mask][0]),np.log(t_rec[mask][-1]),300)

        plt.plot(
            t_line,reg.intercept_ + alpha * t_line,color="red",
            lw=2.5,label=f"Fit α = {alpha:.3f} ({transport})"
        )

    plt.xlabel("log t")
    plt.ylabel("log (MSD - MSD₀)")

    plt.title(
        f"Corrected Transport Classification\n"
        f"→ {transport} "
        f"(Early Time Regime)"
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# SURVIVAL DECAY
# ─────────────────────────────────────────────
def plot_survival_decay(
    t_rec,S_rec,t_validS,log_S,
    late_mask,reg_S,sim_slope,sim_intercept,ana_slope
):

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14, 5))

    # Survival
    ax1.plot(t_rec,S_rec,color="darkorange",lw=2)

    ax1.set_title( "Survival Probability S(t)" )

    ax1.set_xlabel( "Physical time t")

    ax1.set_ylabel("S(t)")

    ax1.set_ylim(bottom=0)

    ax1.grid(True,linestyle='--',alpha=0.6)

    # Log survival
    ax2.plot(t_validS,log_S,color="purple",lw=2,label="Simulation ln(S(t))")

    if late_mask.sum() > 10:

        t_late = t_validS[
            late_mask
        ].reshape(-1, 1)

        ax2.plot(
            t_validS[late_mask],reg_S.predict(t_late),color="cyan",
            lw=2,linestyle='--',label=f"Sim Fit Slope: {sim_slope:.5f}"
        )

        ana_line = (
            ana_slope* t_validS[late_mask]
            + (sim_intercept + (sim_slope - ana_slope)* t_validS[late_mask][0])
        )

        ax2.plot(
            t_validS[late_mask],ana_line,color="red",
            lw=2,linestyle=':',label=f"Analytical Slope: "f"{ana_slope:.5f}"
        )

    ax2.set_title(
        "Log Survival Probability "
        "vs Time (Late Regime)"
    )

    ax2.set_xlabel( "Physical time t" )

    ax2.set_ylabel("ln(S(t))")

    ax2.legend()

    ax2.grid(True,linestyle='--',alpha=0.6)

    plt.tight_layout()
    plt.show()
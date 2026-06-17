import numpy as np
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# TRANSPORT CLASSIFICATION
# ─────────────────────────────────────────────
def classify_transport(t_rec, msd_rec):

    msd_corrected = msd_rec - msd_rec[0]

    mask = ((t_rec >= 1.0) & (t_rec <= 15.0) & (msd_corrected > 0))

    alpha = 0
    transport = "Unknown"

    reg = None

    if mask.sum() >= 10:

        log_t = np.log(t_rec[mask]).reshape(-1, 1)

        log_msd = np.log(msd_corrected[mask])

        reg = LinearRegression().fit(log_t, log_msd)

        alpha = reg.coef_[0]

        if alpha < 0.85:
            transport = "Subdiffusion"

        elif alpha < 1.15:
            transport = "Normal Diffusion"

        elif alpha < 1.85:
            transport = "Superdiffusion"

        else:
            transport = "Ballistic"

    valid_t = ((t_rec > 0) & (msd_corrected > 0))

    return (msd_corrected,mask,alpha,transport,reg,valid_t)


# ─────────────────────────────────────────────
# SURVIVAL DECAY
# ─────────────────────────────────────────────
def analyze_survival_decay(t_rec,S_rec,D,x_lo,x_hi):

    valid_S = S_rec > 1e-12

    t_validS = t_rec[valid_S]

    log_S = np.log(S_rec[valid_S])

    late_mask = t_validS > 20.0

    reg_S = None
    sim_slope = None
    sim_intercept = None
    ana_slope = None

    if late_mask.sum() > 10:

        t_late = t_validS[late_mask].reshape(-1, 1)

        log_S_late = log_S[late_mask]

        reg_S = LinearRegression().fit(t_late,log_S_late)

        sim_slope = reg_S.coef_[0]

        sim_intercept = reg_S.intercept_

        L = x_hi - x_lo

        ana_slope = -2 * D * (np.pi / L)**2

        print(f"\n--- Late Time Decay (t > 20) ---")
        print(f"Simulated Slope:  {sim_slope:.6f}")
        print(f"Analytical Slope: {ana_slope:.6f}")

        print(
            f"Error: "
            f"{abs((sim_slope - ana_slope)/ana_slope)*100:.2f}%"
        )

    return (t_validS,log_S,late_mask,reg_S,sim_slope,sim_intercept,ana_slope)
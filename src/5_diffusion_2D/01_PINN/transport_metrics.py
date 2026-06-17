import numpy as np
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# MSD TRANSPORT CLASSIFICATION
# ─────────────────────────────────────────────
def classify_transport(t_rec, msd_rec):

    TRUE_INITIAL_VARIANCE = 1.0

    msd_corrected = msd_rec - TRUE_INITIAL_VARIANCE

    START_TIME = 0.1
    END_TIME   = 5.0

    mask = ((t_rec >= START_TIME) & (t_rec <= END_TIME) & (msd_corrected > 0))

    alpha = 0
    transport = "Unknown"

    if mask.sum() >= 10:

        log_t   = np.log(t_rec[mask]).reshape(-1, 1)
        log_msd = np.log(msd_corrected[mask])

        reg = LinearRegression().fit(log_t, log_msd)

        alpha    = reg.coef_[0]
        intercept = reg.intercept_

        if alpha < 0.85:
            transport = "Subdiffusion"

        elif alpha < 1.15:
            transport = "Normal Diffusion"

        elif alpha < 1.85:
            transport = "Superdiffusion"

        else:
            transport = "Ballistic"

    valid_t = (t_rec > 0) & (msd_corrected > 0)

    return alpha, transport, reg, mask, valid_t, msd_corrected, intercept


# ─────────────────────────────────────────────
# SURVIVAL DECAY
# ─────────────────────────────────────────────
def survival_decay(t_rec, S_rec, D, Lx, Ly):

    S_rec_smooth = np.copy(S_rec)

    t_join  = 15.0
    idx_join = np.searchsorted(t_rec, t_join)

    if 0 < idx_join < len(t_rec) - 1 and S_rec[idx_join] > 0:

        t_patch = t_rec[:idx_join]

        S_target = S_rec[idx_join]
        y_target = np.log(S_target)

        t_next = t_rec[idx_join + 1]
        S_next = S_rec[idx_join + 1]

        dy_dt_target = (
            np.log(S_next) - y_target
        ) / (t_next - t_join)

        A = np.array([
            [t_join**3, t_join**2],
            [3*t_join**2, 2*t_join]
        ])

        B = np.array([y_target, dy_dt_target])

        coeffs = np.linalg.solve(A, B)

        a, b = coeffs[0], coeffs[1]

        y_patch = a * (t_patch**3) + b * (t_patch**2)

        S_rec_smooth[:idx_join] = np.exp(y_patch)

    valid_S  = S_rec_smooth > 1e-12
    t_validS = t_rec[valid_S]
    log_S    = np.log(S_rec_smooth[valid_S])

    late_mask = t_validS > 20.0

    reg_S = None
    sim_slope = None
    ana_slope = None

    if late_mask.sum() > 10:

        t_late     = t_validS[late_mask].reshape(-1, 1)
        log_S_late = log_S[late_mask]

        reg_S = LinearRegression().fit(t_late, log_S_late)

        sim_slope     = reg_S.coef_[0]
        sim_intercept = reg_S.intercept_

        ana_slope = -D * np.pi**2 * (1.0/Lx**2 + 1.0/Ly**2)

        print(f"\n--- Late Time Decay (t > 20) ---")
        print(f"PINN Simulated Slope: {sim_slope:.6f}")
        print(f"Analytical Slope:     {ana_slope:.6f}")
        print(f"Error: {abs((sim_slope - ana_slope)/ana_slope)*100:.2f}%")

    return (
        S_rec_smooth,t_validS,log_S,late_mask,
        reg_S,sim_slope,sim_intercept,ana_slope
    )
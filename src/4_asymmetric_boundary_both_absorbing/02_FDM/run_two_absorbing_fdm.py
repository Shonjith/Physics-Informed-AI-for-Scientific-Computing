from solver import solve_fdm

from transport_metrics import (
    classify_transport_msd,
    compute_late_time_decay
)

from visualize import (
    plot_solution_snapshots,
    plot_survival_probability,
    plot_msd_classification,
    plot_logS_decay
)

x, snapshots, t_vals, S_vals, msd_vals = solve_fdm()

plot_solution_snapshots(x, snapshots)

plot_survival_probability(t_vals, S_vals)

alpha_exp, classification, reg_msd, mask_msd = classify_transport_msd(t_vals, msd_vals)

plot_msd_classification(t_vals, msd_vals, alpha_exp, classification, reg_msd, mask_msd)

computed_slope, true_slope, reg_s, mask_s = compute_late_time_decay(t_vals, S_vals)

plot_logS_decay(t_vals, S_vals, computed_slope, true_slope, reg_s, mask_s)
from solver import solve_fdm

from transport_metrics import classify_transport

from visualize import (
    plot_solution_snapshots,
    plot_survival_probability,
    plot_transport_classification
)

x, snapshots, t_vals, S_vals = solve_fdm()

plot_solution_snapshots(snapshots)

plot_survival_probability(t_vals, S_vals)

slope, beta, reg, transport_type, mask = classify_transport(t_vals, S_vals)

plot_transport_classification(t_vals, S_vals, slope, beta, reg, mask, transport_type)
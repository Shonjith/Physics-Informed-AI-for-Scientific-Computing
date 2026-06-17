import torch

from model import PINN

from train import train_adam, train_lbfgs

from transport_metrics import (
    compute_transport_metrics,
    classify_transport_msd,
    compute_late_time_decay
)

from visualize import (
    plot_solution_snapshots,
    plot_survival_probability,
    plot_msd_classification,
    plot_logS_decay
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PINN([2, 64, 64, 64, 1]).to(device)

train_adam(model)

train_lbfgs(model)

plot_solution_snapshots(model)

t_vals, S_vals, MSD_vals = compute_transport_metrics(model)

plot_survival_probability(t_vals, S_vals)

alpha, classification, reg_msd, mask_msd = classify_transport_msd(t_vals, MSD_vals)

plot_msd_classification(t_vals, MSD_vals, alpha, classification, reg_msd, mask_msd)

computed_slope, true_slope, reg_s, mask_s = compute_late_time_decay(t_vals, S_vals)

plot_logS_decay(t_vals, S_vals, computed_slope, true_slope, reg_s, mask_s)
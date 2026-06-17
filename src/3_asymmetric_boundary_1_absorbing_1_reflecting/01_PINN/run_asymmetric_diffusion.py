import torch

from model import PINN
from train import pretrain_initial_condition, train_adam, train_lbfgs
from visualize import (
    plot_solution_snapshots,
    plot_survival_probability,
    plot_transport_classification
)
from transport_metrics import compute_survival_curve, classify_transport

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PINN([2, 64, 64, 64, 1]).to(device)

pretrain_initial_condition(model)

train_adam(model)

train_lbfgs(model)

plot_solution_snapshots(model)


t_vals, S_vals = compute_survival_curve(model)

plot_survival_probability(t_vals, S_vals)

slope, beta, reg, transport_type, mask = classify_transport(t_vals, S_vals)

plot_transport_classification(t_vals, S_vals, slope, beta, reg, mask, transport_type)
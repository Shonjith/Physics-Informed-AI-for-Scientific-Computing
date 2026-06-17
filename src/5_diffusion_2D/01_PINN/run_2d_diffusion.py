import torch

from model import PINN2D

from data import device

from train import train_model

from evaluate import evaluate_model

from transport_metrics import (
    classify_transport,
    survival_decay
)

from visualize import (
    plot_heatmaps,
    plot_transport_classification,
    plot_survival_decay
)

print(f"Using device: {device}")

model2d = PINN2D([3, 128, 128, 128, 128, 1]).to(device)

train_model(model2d)

(
    x_arr,y_arr,snap_times,snap_fields,
    t_rec,S_rec,msd_rec,Lx,Ly,D
) = evaluate_model(model2d)

plot_heatmaps(x_arr,y_arr,snap_times,snap_fields)

(
    alpha,transport,reg,
    mask,valid_t,msd_corrected,intercept
) = classify_transport(t_rec, msd_rec)

plot_transport_classification(t_rec, msd_corrected, valid_t, mask, alpha, transport, intercept)

(
    S_rec_smooth,t_validS,log_S,late_mask,
    reg_S,sim_slope,sim_intercept,ana_slope
) = survival_decay(t_rec, S_rec, D, Lx, Ly)

plot_survival_decay(
    t_rec,S_rec_smooth,t_validS,log_S,
    late_mask,reg_S,sim_slope,sim_intercept,ana_slope
)
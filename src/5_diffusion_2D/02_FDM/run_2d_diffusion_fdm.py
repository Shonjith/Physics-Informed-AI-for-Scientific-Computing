from solver import solve_fdm

from transport_metrics import (classify_transport,analyze_survival_decay)

from visualize import (plot_heatmaps,plot_transport_classification,plot_survival_decay)

(
    x_arr,y_arr,snap_times,snap_indices,snap_fields,
    t_rec,S_rec,msd_rec,D,x_lo,x_hi
) = solve_fdm()

plot_heatmaps(x_arr,y_arr,snap_times,snap_indices,snap_fields)

(
    msd_corrected,mask,alpha,transport,
    reg,valid_t
) = classify_transport(t_rec,msd_rec)

plot_transport_classification(t_rec,msd_corrected,valid_t,mask,reg,alpha,transport)

(
    t_validS,log_S,late_mask,reg_S,
    sim_slope,sim_intercept,ana_slope
) = analyze_survival_decay(t_rec,S_rec,D,x_lo,x_hi)

plot_survival_decay(
    t_rec,S_rec,t_validS,log_S,
    late_mask,reg_S,sim_slope,sim_intercept,ana_slope
)
from solver import solve_diffusion
from visualize import *

x, time, U = solve_diffusion()

plot_heatmap(
    x,
    time,
    U
)

plot_time_slices(
    x,
    time,
    U
)
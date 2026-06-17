from model import PINN
from data import device
from train import train_model
from visualize import *

model = PINN().to(device)

model = train_model(model)

plot_heatmap(model)
plot_time_slices(model)
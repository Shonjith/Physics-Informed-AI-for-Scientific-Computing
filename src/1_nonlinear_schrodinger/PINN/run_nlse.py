import torch
from model import PINN
from data import initial_points, collocation_points
from train import train
from visualize import visualize_solution, plot_time_slice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = PINN(layers=[2, 100, 100, 100, 100, 2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

data = {
    "initial": initial_points(50, device),
    "collocation": collocation_points(10000, device)
}

train(model, optimizer, data)

visualize_solution(model, device)

plot_time_slice(model, device, t_value=0.8)


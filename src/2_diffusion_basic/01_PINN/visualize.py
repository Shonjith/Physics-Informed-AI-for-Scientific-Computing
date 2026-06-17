import torch
import matplotlib.pyplot as plt
from data import *

def plot_heatmap(model):

    x = torch.linspace(-L, L, 200)
    t = torch.linspace(0, T, 200)

    X, T_grid = torch.meshgrid(
        x, t,
        indexing='ij'
    )

    x_flat = X.reshape(-1,1).to(device)
    t_flat = T_grid.reshape(-1,1).to(device)

    with torch.no_grad():
        u_pred = model(
            x_flat,
            t_flat
        ).cpu()

    U = u_pred.reshape(200,200).numpy()

    plt.figure(figsize=(8,6))
    plt.imshow(
        U.T,
        extent=[-L, L, 0, T],
        origin='lower',
        aspect='auto'
    )
    plt.colorbar(label="Concentration")
    plt.xlabel("Space (x)")
    plt.ylabel("Time (t)")
    plt.title("Space-Time Heatmap")
    plt.show()


def plot_time_slices(model):

    x = torch.linspace(-L, L, 200)
    times = [0.0, 0.2, 0.5, 1.0]

    plt.figure(figsize=(8,6))

    for time_val in times:

        t_slice = (
            torch.ones_like(x)
            * time_val
        )

        with torch.no_grad():
            u_slice = model(
                x.unsqueeze(1).to(device),
                t_slice.unsqueeze(1).to(device)
            ).cpu()

        plt.plot(
            x.numpy(),
            u_slice.numpy(),
            label=f"t={time_val}"
        )

    plt.xlabel("Space (x)")
    plt.ylabel("Concentration")
    plt.title("Diffusion Over Time")
    plt.legend()
    plt.show()
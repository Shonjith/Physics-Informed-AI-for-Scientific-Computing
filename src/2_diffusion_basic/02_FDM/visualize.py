
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(x, time, U):

    plt.figure(figsize=(8, 6))

    plt.imshow(
        U,
        extent=[
            x.min(),
            x.max(),
            time.min(),
            time.max()
        ],
        origin='lower',
        aspect='auto'
    )

    plt.colorbar(label="Concentration")

    plt.xlabel("Space (x)")
    plt.ylabel("Time (t)")
    plt.title(
        "Explicit FDM: Diffusion Heatmap"
    )

    plt.show()


def plot_time_slices(
    x,
    time,
    U
):

    plt.figure(figsize=(8, 6))

    times_to_plot = [
        0.0,
        0.2,
        0.5,
        1.0
    ]

    for t_val in times_to_plot:

        # Find nearest valid time index
        n_index = np.argmin(
            np.abs(time - t_val)
        )

        plt.plot(
            x,
            U[n_index, :],
            label=f"t={time[n_index]:.3f}"
        )

    plt.xlabel("Space (x)")
    plt.ylabel("Concentration")
    plt.title(
        "Explicit FDM: Time Slices"
    )

    plt.legend()
    plt.show()
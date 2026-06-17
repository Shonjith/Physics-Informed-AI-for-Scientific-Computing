
import numpy as np

# Physical parameters
L = 1.0
T = 1.0
D = 0.1
A = 1.0
sigma = 0.1


def solve_diffusion():

    # Spatial grid
    Nx = 201
    dx = 2 * L / (Nx - 1)

    x = np.linspace(-L, L, Nx)

    # Stability condition
    r = 0.4
    dt = r * dx**2 / D
    Nt = int(T / dt)

    print("dx =", dx)
    print("dt =", dt)
    print("Stability r =", D * dt / dx**2)
    print("Nt =", Nt)

    # Solution matrix
    U = np.zeros((Nt, Nx))

    # Initial condition
    U[0, :] = (
        A * np.exp(
            -x**2 /
            (2 * sigma**2)
        )
    )

    # Time stepping
    for n in range(Nt - 1):

        # Update interior points
        for i in range(1, Nx - 1):

            U[n + 1, i] = (
                U[n, i]
                + r * (
                    U[n, i + 1]
                    - 2 * U[n, i]
                    + U[n, i - 1]
                )
            )

        # Neumann boundary condition
        U[n + 1, 0] = U[n + 1, 1]
        U[n + 1, -1] = U[n + 1, -2]

    # Time array
    time = np.linspace(0, T, Nt)

    return x, time, U
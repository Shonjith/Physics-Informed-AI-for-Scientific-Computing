import numpy as np
import math

L     = 5.0
D     = 0.1
T_max = 900.0

x_left  = -L
x_right = 5 * L

x0    = 0.0
sigma = 1.0

Nx = 301
x  = np.linspace(x_left, x_right, Nx)

dx = x[1] - x[0]

dt = 0.4 * (dx**2) / D
Nt = int(np.ceil(T_max / dt))

dt = T_max / Nt

alpha = D * dt / (dx**2)

def solve_fdm():

    print("Grid Setup:")
    print(f"  dx = {dx:.4f}")
    print(f"  dt = {dt:.4f} (Requires {Nt} steps)")
    print(f"  Stability parameter (alpha) = {alpha:.4f}")

    u = np.exp(-(x - x0)**2 / sigma**2) / (sigma * math.sqrt(math.pi))

    u[0] = 0.0

    snapshot_times = [10.0, 100.0, 300.0, 700.0]
    snapshots = {}

    t_vals = []
    S_vals = []

    save_interval_dt = int(1.0 / dt)

    u_new = np.zeros_like(u)

    print("\nStarting time integration...")

    for n in range(1, Nt + 1):

        current_time = n * dt

        u_new[1:-1] = (
            u[1:-1]
            + alpha * (u[2:] - 2*u[1:-1] + u[:-2])
        )

        u_new[0] = 0.0

        u_new[-1] = u[-1] + alpha * 2 * (u[-2] - u[-1])

        u[:] = u_new[:]

        for t_target in snapshot_times:

            if abs(current_time - t_target) < (dt / 2):
                snapshots[t_target] = u.copy()

        if n % save_interval_dt == 0 or n == Nt:

            t_vals.append(current_time)

            S_vals.append(np.trapz(u, x))

    print("Integration complete.")

    return x, snapshots, np.array(t_vals), np.array(S_vals)
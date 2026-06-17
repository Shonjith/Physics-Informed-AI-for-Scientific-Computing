import numpy as np
import math

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
L       = 5.0
D       = 0.1
T_max   = 900.0

x_left  = -L
x_right = 5 * L

x0      = 0.0
sigma   = 1.0

# ─────────────────────────────────────────────
# SPATIAL AND TEMPORAL GRID
# ─────────────────────────────────────────────
Nx = 301
x  = np.linspace(x_left, x_right, Nx)

dx = x[1] - x[0]

dt_max = (dx**2) / (2 * D)
dt     = dt_max * 0.90

Nt = int(T_max / dt) + 1

alpha = D * dt / (dx**2)


# ─────────────────────────────────────────────
# SOLVER
# ─────────────────────────────────────────────
def solve_fdm():

    print(f"Grid spacing dx: {dx:.4f}")
    print(f"Time step dt: {dt:.5f}")
    print(f"Alpha (stability <= 0.5): {alpha:.4f}\n")

    # Initial condition
    norm = sigma * math.sqrt(math.pi)

    u = np.exp(-(x - x0)**2 / sigma**2) / norm

    # Absorbing boundaries at t=0
    u[0]  = 0.0
    u[-1] = 0.0

    snapshot_times = [10, 100, 300, 700]

    snapshots = {}

    S_vals   = []
    msd_vals = []
    t_vals   = []

    current_time = 0.0

    print("Starting explicit time integration...")

    for n in range(Nt):

        # Survival probability + MSD
        if current_time >= 1.0:

            S_t = np.trapz(u, x)

            S_vals.append(S_t)
            t_vals.append(current_time)

            if S_t > 1e-12:

                u_norm = u / S_t

                msd_t = np.trapz((x - x0)**2 * u_norm, x)

            else:
                msd_t = np.nan

            msd_vals.append(msd_t)

        # Save snapshots
        for t_target in snapshot_times:

            if current_time < t_target <= current_time + dt:
                snapshots[t_target] = u.copy()

        # Explicit FTCS update
        u_next = np.copy(u)

        u_next[1:-1] = ( u[1:-1] + alpha * (u[2:] - 2 * u[1:-1] + u[:-2]))

        # Absorbing boundaries
        u_next[0]  = 0.0
        u_next[-1] = 0.0

        u = u_next

        current_time += dt

    print("Integration complete.\n")

    return (x,snapshots,np.array(t_vals),np.array(S_vals),np.array(msd_vals))
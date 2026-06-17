import numpy as np

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
D = 0.1

x_lo, x_hi = -5.0, 5.0
y_lo, y_hi = -5.0, 5.0

T_max = 50.0

Nx = 200
Ny = 200

dx = (x_hi - x_lo) / (Nx - 1)
dy = (y_hi - y_lo) / (Ny - 1)

r_target = 0.20

dt = r_target * dx**2 / D
r  = D * dt / dx**2

Nt = int(np.ceil(T_max / dt))

x_arr = np.linspace(x_lo, x_hi, Nx)
y_arr = np.linspace(y_lo, y_hi, Ny)

XX, YY = np.meshgrid(x_arr, y_arr, indexing='ij')

# ─────────────────────────────────────────────
# SOLVER
# ─────────────────────────────────────────────
def solve_fdm():

    print(f"Grid:      {Nx} x {Ny}")
    print(f"dx={dx:.4f}  dy={dy:.4f}  dt={dt:.6f}")
    print(f"Time steps:  {Nt}  (T_max={T_max})")

    # Initial condition
    x0, y0, sigma = 0.0, 0.0, 1.0

    u = np.exp( -((XX - x0)**2 + (YY - y0)**2) / sigma**2) / (np.pi * sigma**2)

    # Absorbing boundaries
    u[0, :]  = 0.0
    u[-1, :] = 0.0
    u[:, 0]  = 0.0
    u[:, -1] = 0.0

    # Snapshots
    snap_times   = [0, 1, 5, 10, 25, 50]
    snap_indices = [int(round(t / dt)) for t in snap_times]

    snap_fields = {0: u.copy()}

    # Diagnostics
    r2_arr = (XX**2 + YY**2).ravel()

    dA = dx * dy

    record_every = max(1, Nt // 2000)

    t_rec   = []
    S_rec   = []
    msd_rec = []

    def record(step, u_now):

        u_flat = u_now.ravel()

        S = u_flat.sum() * dA

        if S > 1e-15:
            msd = np.dot(u_flat, r2_arr) * dA / S
        else:
            msd = np.nan

        t_rec.append(step * dt)
        S_rec.append(S)
        msd_rec.append(msd)

    record(0, u)

    # FTCS time stepping
    print("\nRunning FTCS time-stepping ...")

    snap_set = set(snap_indices[1:])

    u = u.copy()

    for n in range(1, Nt + 1):

        lap = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:]
            + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
        ) / dx**2

        u[1:-1, 1:-1] += D * dt * lap

        # Absorbing boundaries
        u[0, :]  = 0.0
        u[-1, :] = 0.0
        u[:, 0]  = 0.0
        u[:, -1] = 0.0

        if n % record_every == 0:
            record(n, u)

        if n in snap_set:
            snap_fields[n] = u.copy()

    record(Nt, u)

    print("Time-stepping complete.")

    t_rec   = np.array(t_rec)
    S_rec   = np.array(S_rec)
    msd_rec = np.array(msd_rec)

    return (
        x_arr,y_arr,snap_times,snap_indices,
        snap_fields,t_rec,S_rec,msd_rec,D,x_lo,x_hi
    )
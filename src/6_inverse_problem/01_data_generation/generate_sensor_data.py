import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(42)

L   = 10.0
T   = 10.0
D   = 0.5

Nx  = 200
Nt  = 50000

dx  = L / (Nx - 1)
dt  = T / Nt

# CFL stability
cfl = D * dt / dx**2

print(
    f"dx = {dx:.4f}  |  "
    f"dt = {dt:.6f}  |  "
    f"CFL = {cfl:.4f}  "
    f"{'✓ STABLE' if cfl < 0.5 else '✗ UNSTABLE — reduce dt!'}"
)

assert cfl < 0.5, ("CFL condition violated — increase Nt!")

x_grid = np.linspace(0, L, Nx)

S_true = np.zeros(Nx)

defect_mask = ((x_grid >= 3.0) & (x_grid <= 4.5))

S_true[defect_mask] = 2.0

print(
    "\nSECRET DEFECT: "
    "S(x) = 2.0 between "
    "x = 3.0 and x = 4.5"
)

print(
    "(The PINN will NOT "
    "be given this information.)\n"
)

sigma0 = 0.8

u = np.exp(-0.5 * ((x_grid - 5.0)/ sigma0)**2)

u[0]  = 0.0
u[-1] = 0.0

N_SAVE = 2000

save_every = max(1,Nt // N_SAVE)

sensor_x = []
sensor_t = []
sensor_u = []

print("Running FDM simulation...")

for n in range(Nt):

    t_now = n * dt

    u_xx = np.zeros(Nx)

    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2

    u_new = u + dt * (D * u_xx - S_true * u)

    u_new[0]  = 0.0
    u_new[-1] = 0.0

    u = u_new

    if n % save_every == 0:

        idx = np.random.randint(1,Nx - 1)

        sensor_x.append(x_grid[idx])

        sensor_t.append(t_now)

        sensor_u.append(u[idx])

sensor_x = np.array(sensor_x)
sensor_t = np.array(sensor_t)
sensor_u = np.array(sensor_u)

noise_level = (0.01 * sensor_u.std())

sensor_u += np.random.normal(0,noise_level,sensor_u.shape)

print(
    f"Collected "
    f"{len(sensor_x)} "
    f"sensor measurements."
)

print(
    f"Noise σ = "
    f"{noise_level:.5f}\n"
)

np.savez("sensor_data.npz",x=sensor_x,t=sensor_t,u=sensor_u)

np.savez("ground_truth_defect.npz",x_grid=x_grid,S_true=S_true)

print("Saved: sensor_data.npz")

print("Saved: ground_truth_defect.npz\n")

fig, axes = plt.subplots(1,3,figsize=(16, 4),facecolor= 'black')

style = dict(facecolor='#0d1117',framealpha=0)

for ax in axes:
    ax.set_facecolor('black')

    ax.tick_params(colors='darkgray')

    ax.xaxis.label.set_color('lightgray')
    ax.yaxis.label.set_color('lightgray')
    ax.title.set_color('whitesmoke')

    for sp in ax.spines.values():
        sp.set_edgecolor('dimgray')

# Plot 1
axes[0].fill_between(x_grid,S_true,alpha=0.3,color='salmon')

axes[0].plot(x_grid,S_true,color='salmon',lw=2)

axes[0].set_xlabel('x')
axes[0].set_ylabel('S(x)')

axes[0].set_title(
    'True Hidden Defect S(x)\n'
    '(PINN must discover this)'
)

axes[0].set_ylim(-0.1, 2.5)

# Plot 2
sc = axes[1].scatter(sensor_x,sensor_t,c=sensor_u,cmap='inferno',s=4,alpha=0.7)

plt.colorbar(sc,ax=axes[1],label='u (temperature)')

axes[1].set_xlabel('x')
axes[1].set_ylabel('t')

axes[1].set_title(
    f'Sensor Measurements '
    f'(n={len(sensor_x)})\n'
    f'(scattered, noisy)'
)

# Plot 3
axes[2].hist(sensor_u,bins=40,color='cornflowerblue',alpha=0.8,edgecolor='dimgray')

axes[2].set_xlabel('u (temperature)')
axes[2].set_ylabel('count')

axes[2].set_title('Sensor Reading Distribution')

plt.tight_layout()

plt.savefig('phase1_data_overview.png',dpi=150,bbox_inches='tight',facecolor='black')

plt.show()

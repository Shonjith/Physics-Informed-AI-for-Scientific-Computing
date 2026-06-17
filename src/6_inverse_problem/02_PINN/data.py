import numpy as np
import torch

# ─────────────────────────────────────────────
# DOMAIN PARAMETERS
# ─────────────────────────────────────────────
x_min = 0.0
x_max = 10.0

t_min = 0.0
t_max = 10.0

def norm_x(x):

    return ((x - x_min) / (x_max - x_min))


def norm_t(t):

    return ((t - t_min) / (t_max - t_min))

def load_sensor_data(device):

    try:
        data = np.load("sensor_data.npz")

    except FileNotFoundError:

        print("Error: 'sensor_data.npz' not found.")

        print("Run 'generate_sensor_data.py' first.")

        exit()

    x_s = data['x'].astype(np.float32)

    t_s = data['t'].astype(np.float32)

    u_s = data['u'].astype(np.float32)

    X_sensor = torch.tensor(np.stack([norm_x(x_s),norm_t(t_s)],axis=1),dtype=torch.float32).to(device)

    U_sensor = torch.tensor(u_s[:, None],dtype=torch.float32).to(device)

    return (X_sensor,U_sensor)

def sample_collocation(n,device):

    xc = np.random.uniform( 0,1,(n, 1)).astype(np.float32)

    tc = np.random.uniform(0,1,(n, 1)).astype(np.float32)

    xc = torch.tensor(xc,dtype=torch.float32).to(device)

    tc = torch.tensor(tc,dtype=torch.float32).to(device)

    return xc, tc
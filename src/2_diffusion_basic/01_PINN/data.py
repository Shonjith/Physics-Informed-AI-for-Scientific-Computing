
import torch

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical parameters
L = 1.0
T = 1.0
D = 0.1
A = 1.0
sigma = 0.1

# PDE interior points
N_pde = 5000
x_pde = (torch.rand(N_pde,1)*(2*L) - L).to(device)
t_pde = (torch.rand(N_pde,1)*T).to(device)

# Initial condition points
N_ic = 1000
x_ic = (torch.rand(N_ic,1)*(2*L) - L).to(device)

# Boundary points
N_bc = 1000
t_bc = (torch.rand(N_bc,1)*T).to(device)

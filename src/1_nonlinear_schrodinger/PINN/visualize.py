import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_solution(model, device):
    model.eval()                       ## Set the model to evaluation mode

    x = np.linspace(-5, 5, 256)
    t = np.linspace(0, np.pi/2, 200)

    T, X = np.meshgrid(t, x)

    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        uv = model(torch.cat([t_flat, x_flat], dim=1))
        u = uv[:, 0].cpu().numpy()
        v = uv[:, 1].cpu().numpy()

    h_abs = np.sqrt(u**2 + v**2)
    h_abs = h_abs.reshape(X.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T, X, h_abs, shading="auto", cmap="viridis")
    plt.colorbar(label=r"$|h(t,x)|$")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("PINN Solution of Nonlinear Schrödinger Equation")
    plt.tight_layout()

    plt.savefig("results/result_spacetime.png", dpi=300)

    plt.show()

def plot_time_slice(model, device, t_value=0.8):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    model.eval()

    x = np.linspace(-5, 5, 400)
    t = np.full_like(x, t_value)

    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        uv = model(torch.cat([t_tensor, x_tensor], dim=1))
        u = uv[:, 0].cpu().numpy()
        v = uv[:, 1].cpu().numpy()

    h_abs = np.sqrt(u**2 + v**2)

    plt.figure(figsize=(8, 5))
    plt.plot(x, h_abs, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("|h(t, x)|")
    plt.title(f"Solution at t = {t_value}")
    plt.grid(True)

    plt.savefig(f"results/time_slice_t_{t_value}.png", dpi=300)

    plt.show()
import torch
from src.physics import schrodinger_residual

def train(model, optimizer, data, epochs=20000):
    t0, x0, u0, v0 = data["initial"]
    tf, xf = data["collocation"]

    for epoch in range(epochs):
        optimizer.zero_grad()

        uv0 = model(torch.cat([t0, x0], dim=1))
        loss_ic = torch.mean((uv0[:,0:1]-u0)**2 + (uv0[:,1:2]-v0)**2)

        f_u, f_v = schrodinger_residual(model, tf, xf)
        loss_pde = torch.mean(f_u**2 + f_v**2)

        loss = loss_ic + loss_pde
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.3e}")

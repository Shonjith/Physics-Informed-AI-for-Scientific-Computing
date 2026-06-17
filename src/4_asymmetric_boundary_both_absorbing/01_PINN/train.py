import torch

from data import sample_points
from physics import loss_function


# ─────────────────────────────────────────────
# PHASE 1: ADAM
# ─────────────────────────────────────────────
def train_adam(model, epochs=20000):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        data = sample_points()

        optimizer.zero_grad()

        loss = loss_function(model, data)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Adam Epoch {epoch:5d} | Loss: {loss.item():.6e}")


# ─────────────────────────────────────────────
# PHASE 2: L-BFGS
# ─────────────────────────────────────────────
def train_lbfgs(model):

    data_lbfgs = sample_points()

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1000,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100
    )

    def closure():

        optimizer.zero_grad()

        loss = loss_function(model, data_lbfgs)

        loss.backward()

        return loss

    optimizer.step(closure)

    print("L-BFGS finished.")
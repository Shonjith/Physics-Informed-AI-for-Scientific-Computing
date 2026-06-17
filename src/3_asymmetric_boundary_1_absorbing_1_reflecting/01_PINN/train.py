import torch

from data import sample_points
from physics import (
    loss_function
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def pretrain_initial_condition(
    model,
    epochs=2000
):
    print(
        "Phase 0: Pre-training Initial Condition..."
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    mse = torch.nn.MSELoss()

    for epoch in range(epochs):

        data = sample_points()

        optimizer.zero_grad()

        x_i = data[5]
        t_i = data[6]
        u_i = data[7]

        u_pred_i = model(
            x_i,
            t_i
        )

        loss = mse(
            u_pred_i,
            u_i
        )

        loss.backward()

        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"Pre-train Epoch {epoch}"
                f" | Loss:"
                f" {loss.item():.6e}"
            )


def train_adam(
    model,
    epochs=10000
):
    print(
        "\nPhase 1: Full Physics Training..."
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    for epoch in range(epochs):

        data = sample_points()

        optimizer.zero_grad()

        loss = loss_function(
            model,
            data
        )

        loss.backward()

        optimizer.step()

        if epoch % 1000 == 0:
            print(
                f"Adam Epoch {epoch}"
                f" | Loss:"
                f" {loss.item():.6e}"
            )


def train_lbfgs(model):

    print(
        "\nPhase 2: L-BFGS..."
    )

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

        loss = loss_function(
            model,
            data_lbfgs
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    print("L-BFGS finished.")
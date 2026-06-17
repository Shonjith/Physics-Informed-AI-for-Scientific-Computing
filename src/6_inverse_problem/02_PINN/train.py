import numpy as np
import torch
import torch.nn as nn
import time

from model import DualPINN

from data import (
    load_sensor_data,
    sample_collocation,
    norm_x
)

from physics import physics_loss

def train_model():

    # Device
    DEVICE = ("cuda" if torch.cuda.is_available()else "cpu")

    print(f"Device: {DEVICE}\n")

    # Hyperparameters
    D = 0.5

    N_COLLOC = 10_000

    N_EPOCHS = 20_000

    LR = 1e-3

    LAMBDA_REG = 0.0005

    # Load sensor data
    (X_sensor,U_sensor) = load_sensor_data(DEVICE)

    # Model
    model = DualPINN().to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(),lr=LR)

    # Evaluation grid
    x_eval_n = torch.tensor(norm_x(np.linspace(0,10,500).astype(np.float32))[:, None]).to(DEVICE)

    print( "Starting training...\n" )

    print(
        f"{'Epoch':>7} | "
        f"{'Loss':>10} | "
        f"{'Data':>10} | "
        f"{'Phys':>10} | "
        f"{'Reg':>8} | "
        f"{'Phys Wt':>7}"
    )

    print("-" * 72)

    t0 = time.time()

    for epoch in range(N_EPOCHS
):

        optimiser.zero_grad()

        # Dynamic physics weight
        lambda_phys = (0.1 if epoch < 5000 else 1.0)

        # Data loss
        u_pred_s = model.forward_u(X_sensor[:, 0:1], X_sensor[:, 1:2])

        l_data = (nn.functional.mse_loss(u_pred_s,U_sensor))

        # Physics loss
        xc_n, tc_n = (sample_collocation(N_COLLOC,DEVICE))

        l_phys = physics_loss(model,xc_n,tc_n,D)

        # Regularization
        S_reg = model.forward_S(x_eval_n)

        l_reg = S_reg.mean()

        # Total loss
        loss = (l_data + (lambda_phys * l_phys) + (LAMBDA_REG * l_reg))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        optimiser.step()

        # Logging
        if (epoch % 1000 == 0 or epoch == N_EPOCHS - 1):

            print(
                f"{epoch:>7} | "
                f"{loss.item():>10.2e} | "
                f"{l_data.item():>10.2e} | "
                f"{l_phys.item():>10.2e} | "
                f"{l_reg.item():>8.4f} | "
                f"{lambda_phys:>7.1f}"
            )

    print(
        f"\nTraining complete "
        f"in "
        f"{time.time()-t0:.1f}s"
    )

    return model
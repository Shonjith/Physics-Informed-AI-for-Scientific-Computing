import torch

from data import sample_points_2d
from physics import loss_function_2d

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_model(model):

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    epochs = 15000

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    loss_history = []

    print("Phase 1: Adam training ...")

    for epoch in range(epochs + 1):

        data = sample_points_2d()

        optimizer.zero_grad()

        total, lf, lb, li = loss_function_2d( model, data )

        total.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        optimizer.step()

        if epoch < epochs:
            scheduler.step()

        loss_history.append(total.item())

        if epoch % 1000 == 0:

            print(
                f"Adam epoch {epoch:5d} "
                f"| total={total.item():.4e} "
                f"| PDE={lf:.4e} "
                f"| BC={lb:.4e} "
                f"| IC={li:.4e} "
                f"| lr={scheduler.get_last_lr()[0]:.2e}"
            )

    # L-BFGS
    print("\nPhase 2: L-BFGS refinement ...")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),lr=0.5,max_iter=50,tolerance_grad=1e-7,
        tolerance_change=1e-9,history_size=100,line_search_fn="strong_wolfe"
    )

    for lbfgs_round in range(50):

        data_lbfgs = sample_points_2d()

        def closure():

            optimizer_lbfgs.zero_grad()

            total, _, _, _ = loss_function_2d(model,data_lbfgs)

            total.backward()

            return total

        loss_val = optimizer_lbfgs.step(closure)

        if lbfgs_round % 5 == 0:
            print(
                f"L-BFGS round "
                f"{lbfgs_round:3d} "
                f"| loss = "
                f"{loss_val.item():.4e}"
            )

    print("\nTraining complete.")
from physics import *

def train_model(model):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 10000

    lambda_pde = 1.0
    lambda_bc = 1.0

    for epoch in range(epochs):

        lambda_ic = 10.0 if epoch < 2000 else 1.0

        optimizer.zero_grad()

        lpde = pde_loss(model, x_pde, t_pde)
        lic = ic_loss(model, x_ic)
        lbc = bc_loss(model, t_bc)

        loss = (
            lambda_pde*lpde +
            lambda_ic*lic +
            lambda_bc*lbc
        )

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch}, "
                f"Loss: {loss.item():.6e}"
            )

    return model
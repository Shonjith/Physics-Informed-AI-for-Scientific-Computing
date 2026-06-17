import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x, t):

        X = torch.cat([x, t], dim=1)

        for layer in self.layers[:-1]:
            X = self.activation(layer(X))

        return self.layers[-1](X)
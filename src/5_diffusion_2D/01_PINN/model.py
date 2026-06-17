import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class PINN2D(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.activation = nn.Tanh()
        self.net = nn.ModuleList()

        for i in range(len(layers) - 1):

            lin = nn.Linear(layers[i], layers[i + 1])

            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

            self.net.append(lin)

    def forward(self, x, y, tau):

        X = torch.cat([x, y, tau], dim=1)

        for layer in self.net[:-1]:
            X = self.activation(layer(X))

        return self.net[-1](X)
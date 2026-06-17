# Two-Dimensional Diffusion
## PINN vs Finite Difference Method

This project investigates diffusion in a bounded two-dimensional domain using both Physics-Informed Neural Networks (PINNs) and Explicit Finite Difference Methods (FDMs).

The study extends the one-dimensional diffusion problem to two spatial dimensions and examines transport behavior through heatmap evolution, mean square displacement (MSD), survival probability, and late-time decay analysis.

---

## Overview

Diffusion in higher dimensions introduces richer spatial dynamics and more complex transport behavior.

A Gaussian probability distribution is initialized at the center of a square domain and evolves according to the two-dimensional diffusion equation under absorbing boundary conditions.

The project compares:

- Physics-Informed Neural Networks (PINNs)
- Explicit Finite Difference Methods (FDMs)

for modeling and analyzing the diffusion process.

---

## Governing Equation

The two-dimensional diffusion equation is

```text
∂u/∂t = D ( ∂²u/∂x² + ∂²u/∂y² )
```

where

- u(x,y,t) is the probability density
- D is the diffusion coefficient
---

## Boundary Conditions

Absorbing boundaries are imposed on all four edges of the square domain.

```text
u(x,y,t) = 0
```

for

```text
x = x_min , x_max
```

and

```text
y = y_min , y_max
```

Particles reaching the boundaries are removed from the system.

---

## Analysis Performed

### Heatmap Evolution

The spatial density distribution is visualized at multiple times to examine how diffusion spreads throughout the domain.

---

## Survival Probability

The total probability remaining inside the domain is

```text
S(t) = ∬ u(x,y,t) dx dy
```

As particles leave through the absorbing boundaries, S(t) decreases with time.
---

## Mean Square Displacement (MSD)

The transport dynamics are characterized using

```text
MSD(t) = ⟨ (x-x₀)² + (y-y₀)² ⟩
```

which measures the spatial spread of the surviving particles.

---

## Transport Classification

Transport behaviour is classified using

```text
MSD(t) ~ t^α
```

where

- α < 1 → Subdiffusion
- α ≈ 1 → Normal Diffusion
- α > 1 → Superdiffusion
---

### Late-Time Survival Decay

The logarithm of the survival probability is analyzed and compared against the theoretical decay rate obtained from the lowest eigenmode of the bounded diffusion problem.

---

## Methods

### PINN

The PINN framework enforces:

- Two-dimensional diffusion equation residual
- Initial condition
- Four absorbing boundary conditions

through automatic differentiation.

The network directly learns the solution over the entire spatio-temporal domain.

---

### Explicit FDM

The FDM implementation uses a two-dimensional FTCS scheme.

The numerical solution serves as a reference for validating the PINN predictions.

---

## Project Structure

```text
5_diffusion_2D
│
├── 01_PINN
│   ├── data.py
│   ├── model.py
│   ├── physics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── transport_metrics.py
│   ├── visualize.py
│   └── run_2d_diffusion.py
│
├── 02_FDM
│   ├── solver.py
│   ├── transport_metrics.py
│   ├── visualize.py
│   └── run_2d_diffusion_fdm.py
│
└── README.md
```

---

## Running the Project

### PINN

```bash
python 01_PINN/run_2d_diffusion.py
```

### FDM

```bash
python 02_FDM/run_2d_diffusion_fdm.py
```

---

## Results

### PINN Heatmap Evolution

![PINN Heatmap](../../results/5_diffusion_2D/01_PINN/heatmap_pinn_24.png)

---

### PINN Survival Probability

![PINN Survival](../../results/5_diffusion_2D/01_PINN/pinn_24_s.png)

---

### PINN MSD Analysis

![PINN MSD](../../results/5_diffusion_2D/01_PINN/pinn_24_msd.png)

---

### FDM Heatmap Evolution

![FDM Heatmap](../../results/5_diffusion_2D/02_FDM/heatmap_fdm_24.png)

---

### FDM Transport Classification

![FDM Log Analysis](../../results/5_diffusion_2D/02_FDM/fdm_24_logs.png)

---

### FDM MSD Analysis

![FDM MSD](../../results/5_diffusion_2D/02_FDM/fdm_24_msd.png)

---

## Key Observations

- The initially localized Gaussian distribution spreads radially throughout the domain.
- Absorbing boundaries continuously remove probability mass from the system.
- Survival probability decreases monotonically with time.
- Mean Square Displacement exhibits diffusive scaling consistent with normal diffusion.
- Late-time survival decay follows the theoretical eigenmode prediction for a bounded square domain.
- PINN and FDM solutions show strong agreement across all transport metrics.

---


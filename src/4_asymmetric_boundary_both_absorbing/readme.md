# Asymmetric Boundary Diffusion
## Both Boundaries Absorbing

This project investigates diffusion in a finite one-dimensional domain with absorbing boundaries at both ends. The study compares Physics-Informed Neural Networks (PINNs) and Explicit Finite Difference Methods (FDMs) for modeling particle transport, survival probability decay, and diffusion classification.

Unlike the mixed boundary case, particles can escape through both ends of the domain, resulting in faster probability loss and distinct transport characteristics.

---

## Overview

The system consists of a finite domain bounded by two absorbing boundaries:

- Left Boundary: Absorbing
- Right Boundary: Absorbing

A Gaussian probability distribution is initialized near the center of the domain and evolves according to the diffusion equation.

The project analyzes:

- Survival probability
- Mean Square Displacement (MSD)
- Transport classification
- Late-time exponential decay
- PINN vs FDM agreement

---

## Governing Equation

The one-dimensional diffusion equation is

```text
∂u/∂t = D ∂²u/∂x²
```

where

- u(x,t) is the probability density
- D is the diffusion coefficient
---

## Boundary Conditions

### Left Boundary

```text
u(x_left,t) = 0
```

### Right Boundary

```text
u(x_right,t) = 0
```

Both boundaries absorb probability density reaching the edges of the domain.

---

## Analysis Performed

## Survival Probability

The total probability remaining inside the domain:

```text
S(t) = ∫ u(x,t) dx
```
As particles leave through both boundaries, \(S(t)\) decreases continuously with time.

---

### Mean Square Displacement (MSD)

The transport dynamics are characterized using

```text
MSD(t) = ⟨(x - x₀)²⟩
```

which measures the spatial spread of the surviving particles.

---

### Transport Classification

Transport behaviour is determined from


```text
MSD(t) ~ t^α
```

- α < 1  → Subdiffusion
- α ≈ 1  → Normal Diffusion
- α > 1  → Superdiffusion

---

### Late-Time Survival Decay

The long-time behaviour of the survival probability is examined through

```text
ln S(t)
```
and compared against the theoretical exponential decay rate.

---

## Methods

### PINN

The Physics-Informed Neural Network enforces:

- Diffusion equation residual
- Initial condition
- Left absorbing boundary
- Right absorbing boundary

through automatic differentiation.

---

### Explicit FDM

The FDM implementation uses the FTCS finite difference scheme with absorbing Dirichlet boundary conditions at both ends of the domain.

The numerical solution serves as a reference for validating PINN predictions.

---

## Project Structure

```text
4_asymmetric_boundary_both_absorbing
│
├── 01_PINN
│   ├── data.py
│   ├── model.py
│   ├── physics.py
│   ├── train.py
│   ├── transport_metrics.py
│   ├── visualize.py
│   └── run_two_absorbing_diffusion.py
│
├── 02_FDM
│   ├── solver.py
│   ├── transport_metrics.py
│   ├── visualize.py
│   └── run_two_absorbing_fdm.py
│
└── README.md
```

---

## Running the Project

### PINN

```bash
python 01_PINN/run_two_absorbing_diffusion.py
```

### FDM

```bash
python 02_FDM/run_two_absorbing_fdm.py
```

---

## Results

### PINN Survival Probability

![PINN Survival](../../results/4_asymmetric_boundary_both_absorbing/01_PINN/pinn_12_s.png)

---

### PINN MSD Analysis

![PINN MSD](../../results/4_asymmetric_boundary_both_absorbing/01_PINN/pinn_12_msd.png)

---

### FDM Survival Probability

![FDM Survival](../../results/4_asymmetric_boundary_both_absorbing/02_FDM/fdm_12_s.png)

---

### FDM MSD Analysis

![FDM MSD](../../results/4_asymmetric_boundary_both_absorbing/02_FDM/fdm_12_msd.png)

---

### PINN vs FDM Comparison

![Comparison](../../results/4_asymmetric_boundary_both_absorbing/03_comparison/fdm_pinn_12.png)

---

## Key Observations

- Probability decreases more rapidly than the mixed boundary case because particles can escape through both ends.
- Mean Square Displacement initially follows normal diffusive behaviour.
- Survival probability exhibits exponential decay at late times.
- The decay rate agrees with theoretical expectations for a finite absorbing domain.
- PINN predictions closely match the FDM reference solution for both survival probability and transport metrics.

---


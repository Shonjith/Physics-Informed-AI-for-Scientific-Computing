# PINN vs FDM for Scientific Computing

A comparative study of **Physics-Informed Neural Networks (PINNs)** and **Finite Difference Methods (FDMs)** for solving partial differential equations and investigating transport phenomena in diffusion systems.

This repository contains implementations of PINNs and classical numerical solvers across a variety of physical systems, including diffusion equations, nonlinear Schrödinger equations, asymmetric boundary conditions, two-dimensional transport, and inverse problems.

---

## Overview

Physics-Informed Neural Networks (PINNs) provide a mesh-free framework for solving differential equations by embedding physical laws directly into the loss function of neural networks. While traditional numerical methods such as Finite Difference Methods (FDMs) are well established and computationally efficient, PINNs offer flexibility for handling complex geometries, inverse problems, and sparse observational data.

This repository investigates the strengths and limitations of both approaches through several benchmark problems.

---

## Repository Structure

```text
PINN_VS_FDM_SCIENTIFIC_COMPUTING
│
├── report/
│
├── results/
│   ├── 1_nonlinear_schrodinger/
│   ├── 2_diffusion_basic/
│   ├── 3_asymmetric_boundary_1_absorbing_1_reflecting/
│   ├── 4_asymmetric_boundary_both_absorbing/
│   ├── 5_diffusion_2D/
│   └── 6_inverse_problem/
│
├── src/
│   ├── 1_nonlinear_schrodinger/
│   ├── 2_diffusion_basic/
│   ├── 3_asymmetric_boundary_1_absorbing_1_reflecting/
│   ├── 4_asymmetric_boundary_both_absorbing/
│   ├── 5_diffusion_2D/
│   └── 6_inverse_problem/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Projects Included

### 1. Nonlinear Schrödinger Equation

Implementation of a Physics-Informed Neural Network for solving the one-dimensional Nonlinear Schrödinger Equation.

**Objectives**
- Learn complex wave dynamics using PINNs
- Compare PINN predictions with analytical solutions
- Investigate nonlinear wave propagation

---

### 2. Basic Diffusion Equation

Comparison of PINN and Explicit Finite Difference Method for the one-dimensional diffusion equation.

**Objectives**
- Validate PINN performance against a classical numerical solver
- Analyze accuracy and convergence
- Study diffusion dynamics under standard boundary conditions

---

### 3. Asymmetric Boundary Diffusion  
#### One Absorbing Boundary + One Reflecting Boundary

Diffusion in a finite domain with mixed boundary conditions.

**Analysis Performed**
- Solution snapshots
- Survival probability
- Log-log transport classification
- Late-time decay behaviour

---

### 4. Asymmetric Boundary Diffusion  
#### Both Boundaries Absorbing

Diffusion in a finite domain with absorbing boundaries on both sides.

**Analysis Performed**
- Survival probability
- Mean Square Displacement (MSD)
- Transport classification
- Exponential survival decay
- Comparison with theoretical predictions

---

### 5. Two-Dimensional Diffusion

Two-dimensional diffusion in a bounded square domain.

**Analysis Performed**
- Density heatmaps
- Mean Square Displacement
- Transport classification
- Survival probability
- Late-time decay analysis
- Comparison with analytical eigenmode predictions

---

### 6. Inverse Problem: Hidden Defect Discovery

Inverse PINN framework for discovering an unknown spatial defect profile from sparse sensor measurements.

**Objectives**
- Recover hidden source/sink terms
- Train using noisy sensor observations
- Solve an inverse PDE problem without prior knowledge of the defect location

**Outputs**
- Sensor data generation
- PINN reconstruction
- Defect localization
- Comparison with ground truth

---

## Methodology

### Physics-Informed Neural Networks (PINNs)

PINNs approximate the solution of a PDE using a neural network while enforcing:

- Governing differential equations
- Initial conditions
- Boundary conditions

through automatic differentiation and physics-based loss functions.

### Finite Difference Methods (FDM)

Explicit finite difference schemes are used as benchmark solvers for diffusion problems.

These implementations provide:

- Numerical reference solutions
- Validation of PINN predictions
- Quantitative performance comparisons

---

## Key Topics Covered

- Physics-Informed Neural Networks
- Scientific Machine Learning
- Partial Differential Equations
- Diffusion Processes
- Nonlinear Schrödinger Equation
- Survival Probability Analysis
- Mean Square Displacement (MSD)
- Transport Classification
- Inverse Problems
- Scientific Computing

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shonjith/Physics-Informed-AI-for-Scientific-Computing.git
cd Physics-Informed-AI-for-Scientific-Computing
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Results

Simulation outputs, trained models, plots, and comparison figures are available inside:

```text
results/
```

Each project folder contains its own outputs and visualizations.

---

## Report

The complete project report and detailed analysis can be found in:

```text
report/
```

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-Learn

---


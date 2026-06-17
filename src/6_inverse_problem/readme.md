# Hidden Defect Discovery using Physics-Informed Neural Networks (PINNs)

## Overview

This project demonstrates the use of Physics-Informed Neural Networks (PINNs) for solving an inverse diffusion problem. A hidden spatial defect profile is embedded inside a one-dimensional diffusion equation and synthetic sensor measurements are generated using an Explicit Finite Difference Method (FDM). Using only sparse and noisy sensor observations, the PINN simultaneously reconstructs the temperature field and discovers the unknown defect profile.

The objective is to investigate whether physics-informed learning can recover hidden physical properties from indirect observations without direct access to the underlying defect distribution.

---

## Problem Statement

Consider the diffusion-reaction equation

[
u_t = D u_{xx} - S(x)u
]

where:

* (u(x,t)) is the temperature field
* (D) is the diffusion coefficient
* (S(x)) is an unknown spatial defect profile

The defect profile acts as a local sink term that removes energy from the system. The challenge is to recover (S(x)) using only scattered temperature measurements.

---

## Repository Structure

```text
05_inverse_pinn_hidden_defect/
│
├── README.md
│
├── 01_data_generation/
│   └── generate_sensor_data.py
│
├── 02_pinn/
│   ├── model.py
│   ├── data.py
│   ├── physics.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_inverse_pinn.py
```

---

## Physics Model

### Forward Problem

Synthetic measurements are generated using an Explicit Finite Difference Method (FDM):

[
u_t = D u_{xx} - S(x)u
]

with:

* Diffusion coefficient: (D = 0.5)
* Rod length: (L = 10)
* Gaussian initial condition centered at (x=5)
* Dirichlet boundary conditions (u(0,t)=u(L,t)=0)

### Inverse Problem

The PINN learns:

1. Temperature field (u(x,t))
2. Hidden defect profile (S(x))

simultaneously from sparse sensor measurements and the governing PDE.

---

## Workflow

### Phase 1: Data Generation

A synthetic diffusion simulation is performed using FDM.

* Hidden defect profile is inserted into the domain.
* Approximately 2000 noisy sensor measurements are collected.
* Sensor measurements are saved for PINN training.

### Phase 2: PINN Training

A dual-network architecture is used:

* Network 1 predicts (u(x,t))
* Network 2 predicts (S(x))

Training minimizes:

* Sensor data loss
* PDE residual loss
* Defect regularization loss

### Phase 3: Evaluation

The learned defect profile is compared with the ground truth profile and visualized.

---

## Results

The PINN successfully identifies:

* Location of the hidden defect
* Approximate defect width
* Defect intensity profile

Example result:

* True defect region: (3.0 \le x \le 4.5)
* PINN predicted defect region closely matches the ground truth.

---

## Installation

```bash
git clone <repository-url>
cd 6_inverse_problem

pip install numpy matplotlib torch
```

---

## Usage

### Generate Sensor Data

```bash
python 01_data_generation/generate_sensor_data.py
```

### Train and Evaluate PINN

```bash
python 02_pinn/run_inverse_pinn.py
```

---

## Future Work

* Multiple hidden defects
* Two-dimensional inverse diffusion problems
* Bayesian uncertainty quantification
* Real experimental sensor measurements

---



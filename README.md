# Riemannian Momentum Matching on SPD Manifolds

This repository contains the official PyTorch implementation for the paper:

**"Riemannian Momentum Matching on SPD Manifolds: A Unified PDE and Geometric Framework with Applications to Radar Signal Processing"**
*Submitted to IEEE Transactions on Signal Processing, 2026*

### Authors
**Ahmed Hasan Mohmed Abaker**¹
¹Department of Mathematics, University of Juba, Juba, South Sudan
Email: abkra380@gmail.com

### Abstract
We propose Riemannian Momentum Matching (RMM), a unified theoretical framework that connects Riemannian optimization on Symmetric Positive Definite (SPD) manifolds with Partial Differential Equations. We prove that Riemannian SGD corresponds to the Heat Equation and Riemannian Momentum corresponds to a damped Wave Equation. The proposed RMM algorithm achieves accelerated convergence rates and state-of-the-art performance on three radar signal processing tasks: MSTAR Target Recognition, STAP Clutter Suppression, and DoA Estimation.

### Key Contributions
- **Unified PDE Theory**: Theoretical connection between RSGD/Heat Flow and R-Momentum/Wave Equation
- **RMM Algorithm**: Symplectic integrator with Center Manifold convergence analysis: $O((1-\sqrt{\mu/L})^t)$
- **Radar Applications**: 2.5x faster convergence vs RSGD with 10% computational overhead

### Installation
```bash
git clone https://github.com/YourUsername/RMM-SPD.git
cd RMM-SPD
pip install -r requirements.txt

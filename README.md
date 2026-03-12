# Quantum Rodeo Algorithm

This repository contains the source code, simulation scripts, and LaTeX report for the implementation of the **Rodeo Algorithm** using Qiskit.

## Overview
The Rodeo Algorithm is a stochastic quantum filter used for eigenvalue estimation. This project applies the algorithm to:
1. The 1D Transverse-Field Ising Model ($L=4$)
2. The 1D Naive Discrete Dirac Hamiltonian 
3. The 1D Heisenberg XXX Spin Chain ($L=3$)

## Contents
- `rodeo_noiseless.py`: Ideal Qiskit simulation of the Ising model rodeo filter.
- `rodeo_noisy.py`: Hardware-optimized single-ancilla Qiskit implementation with custom depolarising and IBM Manila noise profile simulations.
- `rodeo_dirac.py`: Generalization of the rodeo filter to the 1D discrete Dirac Hamiltonian.
- `generate_*.py`: Standalone Python scripts used to generate the theoretical eigenvalue plots and algorithmic reconstruction validations.
- `main.tex`: The LaTeX source code for the 5-page technical report detailing the mathematical framework, Qiskit implementation, noise analysis, and results.
- `*.png`: Simulation output plots embedded in the report.

## Requirements
- Python 3.12+
- `qiskit`
- `qiskit-aer`
- `numpy`
- `matplotlib`
- `tqdm`

## Context
This work was authored by Murilo Pedroso and Arthur Felipe Cavalcante de Souza Mello (CentraleSupélec) under the advisement of Brice Chichereau and Philippe Deniel (CEA, France).

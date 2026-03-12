"""
Rodeo Algorithm — Noiseless Simulation for the Transverse-Field Ising Model
Authors: Murilo, Arthur
Reference: Choi et al., arXiv:2009.04092

This script implements the Rodeo Algorithm to estimate the energy spectrum
of the 1D Transverse-Field Ising Model (TFIM) Hamiltonian:

    H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i

The rodeo filter works by repeatedly applying controlled time-evolution
U(t) = exp(-iHt) gated on ancilla qubits prepared in superposition.
After each cycle the ancilla is measured; its success probability is:

    P(E) = prod_{k=1}^{N} cos^2[(E_n - E_target) t_k / 2]

Peaks in P(E) vs E_target reveal the true eigenvalues E_n.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
#  Rodeo circuit builder
# ---------------------------------------------------------------------------
def create_rodeo_circuit(t_list, E_target, J, h, num_qubits, trotter_steps=10):
    """
    Build a full Rodeo Algorithm circuit with N = len(t_list) ancilla cycles.

    Parameters
    ----------
    t_list : array-like
        Random evolution times for each rodeo cycle.
    E_target : float
        Energy value to probe.
    J : float
        Ising coupling constant.
    h : float
        Transverse field strength.
    num_qubits : int
        Number of system (spin) qubits.
    trotter_steps : int
        Number of Trotter slices per cycle for exp(-iHt).

    Returns
    -------
    QuantumCircuit
        The assembled rodeo circuit with measurements on all ancillae.
    """
    cycles = len(t_list)
    qr_sys = QuantumRegister(num_qubits, "sys")
    qr_anc = QuantumRegister(cycles, "anc")
    cr_anc = ClassicalRegister(cycles, "meas")
    qc = QuantumCircuit(qr_sys, qr_anc, cr_anc)

    for m in range(cycles):
        t = t_list[m]
        dt = t / trotter_steps

        # --- Step 1-2: Prepare ancilla in |+> via X then H ---
        qc.x(qr_anc[m])
        qc.h(qr_anc[m])

        # --- Step 3: Controlled Trotterised evolution exp(-iHt) ---
        for _ in range(trotter_steps):
            # Transverse-field term:  -h * X_i
            for i in range(num_qubits):
                qc.crx(-2 * h * dt, qr_anc[m], qr_sys[i])
            # Ising interaction term: -J * Z_i Z_{i+1}
            for i in range(num_qubits - 1):
                qc.cx(qr_sys[i], qr_sys[i + 1])
                qc.crz(-2 * J * dt, qr_anc[m], qr_sys[i + 1])
                qc.cx(qr_sys[i], qr_sys[i + 1])

        # --- Step 4: Phase kick  exp(i E_target t) on ancilla ---
        qc.p(E_target * t, qr_anc[m])

        # --- Step 5-6: Hadamard and measurement ---
        qc.h(qr_anc[m])
        qc.measure(qr_anc[m], cr_anc[m])

    return qc


# ---------------------------------------------------------------------------
#  Simulation parameters
# ---------------------------------------------------------------------------
J_COUPLING   = 1.0      # Ising coupling
H_FIELD      = 1.5      # Transverse field
NUM_QUBITS   = 4        # Number of spin sites (L)
CYCLES       = 10       # Number of rodeo cycles (ancilla qubits)
SHOTS        = 20000    # Measurement shots per energy point
T_RMS        = 5.0      # RMS width of Gaussian random times
SCAN_POINTS  = 300      # Resolution of the energy scan
E_MIN, E_MAX = -8, 8    # Energy scan window

# Fix random seed for reproducibility
np.random.seed(42)
t_list = np.random.normal(0, T_RMS, CYCLES)

# Energy grid
E_scan = np.linspace(E_MIN, E_MAX, SCAN_POINTS)

# ---------------------------------------------------------------------------
#  Ideal (noiseless) backend
# ---------------------------------------------------------------------------
backend = AerSimulator()
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

# ---------------------------------------------------------------------------
#  Energy sweep
# ---------------------------------------------------------------------------
success_prob = []
success_str = "1" * CYCLES   # All ancillae measured |1>

print(f"Rodeo noiseless scan: {SCAN_POINTS} points, {SHOTS} shots, {CYCLES} cycles")
for E in tqdm(E_scan, desc="Energy sweep"):
    qc = create_rodeo_circuit(t_list, E, J_COUPLING, H_FIELD, NUM_QUBITS)
    transpiled = pm.run(qc)
    counts = backend.run(transpiled, shots=SHOTS).result().get_counts()
    success_prob.append(counts.get(success_str, 0) / SHOTS)

# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(E_scan, success_prob, "b-o", markersize=3, linewidth=1.2,
         label=f"Qiskit signal ({CYCLES} cycles)")
plt.xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
plt.ylabel(r"Success probability $P(N)$", fontsize=13)
plt.title(f"Simulated Spectrum — Ising Model  $L={NUM_QUBITS}$  (noiseless)",
          fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("spectrum_noiseless.png", dpi=200)
plt.show()
print("Saved: spectrum_noiseless.png")

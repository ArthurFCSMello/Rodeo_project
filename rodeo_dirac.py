"""
Rodeo Algorithm — Generalised to the 1D Discrete Dirac Hamiltonian
Authors: Murilo, Arthur

The lattice (Wilson–)Dirac Hamiltonian in 1+1 dimensions reads:

    H_D = (1/2a) sum_n  sigma_z (|n+1><n| - |n-1><n|)  +  m * sigma_x

where  a  is the lattice spacing,  m  the fermion mass, and  sigma_{x,z}
are Pauli matrices acting on a two-component spinor (chirality) index.

We encode the position index  n = 0,...,L-1  in  q = log2(L)  qubits
and the spinor component in 1 additional qubit, giving  q+1  system
qubits.  The hopping (shift) operator is implemented via an increment
circuit.

The Rodeo filter is identical to the Ising case: controlled exp(-iH_D t)
followed by ancilla phase and measurement.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
#  Increment gate:  |n> -> |n+1 mod 2^q>   (used to build the shift)
# ---------------------------------------------------------------------------
def increment_gate(qr, qc):
    """Apply +1 (mod 2^q) on register qr inside circuit qc."""
    n = len(qr)
    for i in range(n - 1, 0, -1):
        qc.mcx(list(qr[:i]), qr[i])  # multi-controlled NOT
    qc.x(qr[0])


def decrement_gate(qr, qc):
    """Apply -1 (mod 2^q) on register qr inside circuit qc."""
    n = len(qr)
    qc.x(qr[0])
    for i in range(1, n):
        qc.mcx(list(qr[:i]), qr[i])


# ---------------------------------------------------------------------------
#  Controlled Trotter step for H_Dirac
# ---------------------------------------------------------------------------
def controlled_dirac_trotter(qc, qr_pos, qr_spin, qr_anc, dt, mass, a=1.0):
    """
    One controlled Trotter slice of the 1D Dirac Hamiltonian.

    H_D  =  (1/2a) sigma_z T  +  m sigma_x

    T is the discrete derivative (shift-right minus shift-left).
    We split into  exp(-i dt H_kin) * exp(-i dt H_mass).
    """
    # ---- Kinetic term: sigma_z * (shift_right) ----
    # Controlled rotation: first apply controlled-Z on spin qubit
    # then controlled-increment on position register.
    # For the Trotter slice approximation we use:
    #   exp(-i dt/(2a) sigma_z T) ≈ controlled_phase * controlled_shift
    hop = dt / (2.0 * a)

    # Forward hopping controlled by ancilla
    qc.crz(-2 * hop, qr_anc[0], qr_spin[0])   # sigma_z phase
    # Controlled increment on position register
    ctrl_list = [qr_anc[0]] + list(qr_pos[:-1])
    qc.mcx(ctrl_list, qr_pos[-1])
    for i in range(len(qr_pos) - 2, 0, -1):
        ctrl = [qr_anc[0]] + list(qr_pos[:i])
        qc.mcx(ctrl, qr_pos[i])
    qc.cx(qr_anc[0], qr_pos[0])

    # Backward hopping (reverse shift)
    qc.crz(2 * hop, qr_anc[0], qr_spin[0])
    qc.cx(qr_anc[0], qr_pos[0])
    for i in range(1, len(qr_pos)):
        ctrl = [qr_anc[0]] + list(qr_pos[:i])
        qc.mcx(ctrl, qr_pos[i])

    # ---- Mass term: m * sigma_x ----
    qc.crx(-2 * mass * dt, qr_anc[0], qr_spin[0])


# ---------------------------------------------------------------------------
#  Full Rodeo circuit for the Dirac Hamiltonian
# ---------------------------------------------------------------------------
def create_dirac_rodeo_circuit(t_list, E_target, mass, a,
                                num_pos_qubits, trotter_steps=6):
    """
    Rodeo circuit targeting the 1D discrete Dirac Hamiltonian.

    Parameters
    ----------
    num_pos_qubits : int
        Number of qubits encoding position (L = 2^num_pos_qubits sites).
    mass : float
        Fermion mass  m.
    a : float
        Lattice spacing.
    """
    cycles = len(t_list)
    qr_pos  = QuantumRegister(num_pos_qubits, "pos")
    qr_spin = QuantumRegister(1, "spin")
    qr_anc  = QuantumRegister(cycles, "anc")
    cr_anc  = ClassicalRegister(cycles, "meas")

    qc = QuantumCircuit(qr_pos, qr_spin, qr_anc, cr_anc)

    for m_cycle in range(cycles):
        t = t_list[m_cycle]
        dt = t / trotter_steps

        qc.x(qr_anc[m_cycle])
        qc.h(qr_anc[m_cycle])

        for _ in range(trotter_steps):
            controlled_dirac_trotter(
                qc, qr_pos, qr_spin,
                QuantumRegister(bits=[qr_anc[m_cycle]]),
                dt, mass, a)

        qc.p(E_target * t, qr_anc[m_cycle])
        qc.h(qr_anc[m_cycle])
        qc.measure(qr_anc[m_cycle], cr_anc[m_cycle])

    return qc


# ---------------------------------------------------------------------------
#  Simulation parameters
# ---------------------------------------------------------------------------
MASS         = 1.0       # Fermion mass
A_SPACING    = 1.0       # Lattice spacing
POS_QUBITS   = 2         # 2^2 = 4 lattice sites
CYCLES       = 6         # Rodeo cycles
SHOTS        = 10000
T_RMS        = 4.0
SCAN_POINTS  = 120
E_MIN, E_MAX = -4, 4
TROTTER      = 6

np.random.seed(123)
t_list = np.random.normal(0, T_RMS, CYCLES)
E_scan = np.linspace(E_MIN, E_MAX, SCAN_POINTS)

# ---------------------------------------------------------------------------
#  Run
# ---------------------------------------------------------------------------
backend = AerSimulator()
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

success_prob = []
success_str = "1" * CYCLES

print(f"Dirac Rodeo scan: L={2**POS_QUBITS} sites, mass={MASS}, "
      f"{SCAN_POINTS} points, {SHOTS} shots")

for E in tqdm(E_scan, desc="Dirac energy sweep"):
    qc = create_dirac_rodeo_circuit(
        t_list, E, MASS, A_SPACING, POS_QUBITS, TROTTER)
    transpiled = pm.run(qc)
    counts = backend.run(transpiled, shots=SHOTS).result().get_counts()
    success_prob.append(counts.get(success_str, 0) / SHOTS)

# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(E_scan, success_prob, "m-o", markersize=3, linewidth=1.2,
         label=f"Dirac spectrum ({CYCLES} cycles)")
plt.xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
plt.ylabel(r"Success probability $P(N)$", fontsize=13)
plt.title(r"Rodeo Spectrum — 1D Discrete Dirac  "
          rf"$L={2**POS_QUBITS},\; m={MASS}$", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("spectrum_dirac.png", dpi=200)
plt.show()
print("Saved: spectrum_dirac.png")

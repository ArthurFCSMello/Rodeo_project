"""
Rodeo Algorithm — Noisy Simulations for the Transverse-Field Ising Model
Authors: Murilo, Arthur
Reference: Choi et al., arXiv:2009.04092

Two noise models are compared:
  (A) Custom depolarising noise  (p_1q = 0.1%, p_2q = 1%)
  (B) Realistic IBM Manila noise  (FakeManilaV2 backend snapshot)

Both use a single-ancilla sequential scheme with mid-circuit measurement
and conditional reset, reducing qubit count at the cost of depth.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel, depolarizing_error
from tqdm.auto import tqdm

# Try importing the fake backend (requires qiskit-ibm-runtime)
try:
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    HAS_MANILA = True
except ImportError:
    HAS_MANILA = False
    print("Warning: qiskit-ibm-runtime not found; skipping Manila panel.")


# ---------------------------------------------------------------------------
#  Multi-ancilla Rodeo circuit  (used with depolarising noise)
# ---------------------------------------------------------------------------
def create_rodeo_multi_ancilla(t_list, E_target, J, h, num_qubits,
                                trotter_steps=10):
    """Multi-ancilla Rodeo circuit — one ancilla per cycle."""
    cycles = len(t_list)
    qr_sys = QuantumRegister(num_qubits, "sys")
    qr_anc = QuantumRegister(cycles, "anc")
    cr_anc = ClassicalRegister(cycles, "meas")
    qc = QuantumCircuit(qr_sys, qr_anc, cr_anc)

    for m in range(cycles):
        t, dt = t_list[m], t_list[m] / trotter_steps
        qc.x(qr_anc[m]);  qc.h(qr_anc[m])

        for _ in range(trotter_steps):
            for i in range(num_qubits):
                qc.crx(-2 * h * dt, qr_anc[m], qr_sys[i])
            for i in range(num_qubits - 1):
                qc.cx(qr_sys[i], qr_sys[i + 1])
                qc.crz(-2 * J * dt, qr_anc[m], qr_sys[i + 1])
                qc.cx(qr_sys[i], qr_sys[i + 1])

        qc.p(E_target * t, qr_anc[m])
        qc.h(qr_anc[m])
        qc.measure(qr_anc[m], cr_anc[m])

    return qc


# ---------------------------------------------------------------------------
#  Single-ancilla Rodeo circuit  (used with Manila noise)
# ---------------------------------------------------------------------------
def create_rodeo_single_ancilla(t_list, E_target, J, h, num_qubits,
                                 trotter_steps=10):
    """
    Single-ancilla Rodeo circuit with mid-circuit measurement and
    conditional reset.  Reduces qubit count for hardware compatibility.
    """
    cycles = len(t_list)
    qr_sys = QuantumRegister(num_qubits, "sys")
    qr_anc = QuantumRegister(1, "anc")
    cr_anc = ClassicalRegister(1, "meas")
    qc = QuantumCircuit(qr_sys, qr_anc, cr_anc)

    qc.x(qr_anc)  # Start ancilla in |1>

    for m in range(cycles):
        t, dt = t_list[m], t_list[m] / trotter_steps
        qc.h(qr_anc)

        for _ in range(trotter_steps):
            for i in range(num_qubits):
                qc.crx(-2 * h * dt, qr_anc, qr_sys[i])
            for i in range(num_qubits - 1):
                qc.cx(qr_sys[i], qr_sys[i + 1])
                qc.crz(-2 * J * dt, qr_anc, qr_sys[i + 1])
                qc.cx(qr_sys[i], qr_sys[i + 1])

        qc.p(E_target * t, qr_anc)
        qc.h(qr_anc)
        qc.measure(qr_anc, cr_anc)

        # Conditional reset: undo phase and flip back if measured |1>
        qc.p(-E_target * t, qr_anc)
        qc.x(qr_anc)

    return qc


# ---------------------------------------------------------------------------
#  Simulation parameters
# ---------------------------------------------------------------------------
J_COUPLING   = 1.0
H_FIELD      = 1.5
NUM_QUBITS   = 4
CYCLES       = 10
SHOTS        = 1000
T_RMS        = 5.0
SCAN_POINTS  = 50
E_MIN, E_MAX = -8, 8
TROTTER      = 20

np.random.seed(42)
t_list = np.random.normal(0, T_RMS, CYCLES)
E_scan = np.linspace(E_MIN, E_MAX, SCAN_POINTS)


# ---------------------------------------------------------------------------
#  (A)  Depolarising noise model
# ---------------------------------------------------------------------------
def run_depolarising_scan():
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.001, 1), ["x", "h", "p"])
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.01,  2), ["cx", "crx", "crz"])

    backend = AerSimulator(noise_model=noise_model)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    probs = []
    success_str = "1" * CYCLES
    for E in tqdm(E_scan, desc="Depolarising scan"):
        qc = create_rodeo_multi_ancilla(
            t_list, E, J_COUPLING, H_FIELD, NUM_QUBITS, TROTTER)
        counts = backend.run(pm.run(qc), shots=SHOTS).result().get_counts()
        probs.append(counts.get(success_str, 0) / SHOTS)
    return probs


# ---------------------------------------------------------------------------
#  (B)  IBM Manila realistic noise model
# ---------------------------------------------------------------------------
def run_manila_scan():
    if not HAS_MANILA:
        return None
    machine = FakeManilaV2()
    noise_model = NoiseModel.from_backend(machine)
    backend = AerSimulator(noise_model=noise_model,
                           coupling_map=machine.coupling_map)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    probs = []
    for E in tqdm(E_scan, desc="IBM Manila scan"):
        qc = create_rodeo_single_ancilla(
            t_list, E, J_COUPLING, H_FIELD, NUM_QUBITS, TROTTER)
        counts = backend.run(pm.run(qc), shots=SHOTS).result().get_counts()
        probs.append(counts.get("1", 0) / SHOTS)
    return probs


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prob_depol = run_depolarising_scan()
    prob_manila = run_manila_scan()

    n_panels = 2 if prob_manila is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(E_scan, prob_depol, "r-o", markersize=3, linewidth=1.2)
    axes[0].set_title("Depolarising noise", fontsize=13)
    axes[0].set_xlabel(r"$E_{\mathrm{target}}$")
    axes[0].set_ylabel(r"$P(N)$")
    axes[0].grid(True, alpha=0.3)

    if prob_manila is not None:
        axes[1].plot(E_scan, prob_manila, "g-o", markersize=3, linewidth=1.2)
        axes[1].set_title("IBM Manila noise", fontsize=13)
        axes[1].set_xlabel(r"$E_{\mathrm{target}}$")
        axes[1].set_ylabel(r"$P(N)$")
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Noisy Rodeo Spectrum — Ising $L={NUM_QUBITS}$", fontsize=14)
    plt.tight_layout()
    plt.savefig("spectrum_noisy_comparison.png", dpi=200)
    plt.show()
    print("Saved: spectrum_noisy_comparison.png")

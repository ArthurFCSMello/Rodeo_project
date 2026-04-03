"""
Rodeo Algorithm — Noisy Simulations for the Transverse-Field Ising Model
Authors: Murilo, Arthur
Reference: Choi et al., arXiv:2009.04092

Two noise models are applied in the IBM real hardware and compared :
    1. Raw (No error suppression)
    2. Mitigated (With DD and Pauli Twirling)

Both use a single-ancilla sequential scheme with mid-circuit measurement
and reset, reducing qubit count at the cost of depth.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel, depolarizing_error
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

from qiskit_ibm_runtime import QiskitRuntimeService

# Your 44-character token here
MY_API_TOKEN = "TOKEN"
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=MY_API_TOKEN)

print("Authenticating with IBM Quantum Platform...")

# Using the EXACT string demanded by your qiskit-ibm-runtime version
service = QiskitRuntimeService(
    channel="ibm_quantum_platform", 
    token=MY_API_TOKEN
)

print("Authentication successful! Fetching backend...")

real_backend = service.least_busy(operational=True, simulator=False, min_num_qubits=5)

print(f"🎉 Successfully connected to real quantum hardware: {real_backend.name}")
print(f"Number of qubits: {real_backend.num_qubits}")

# --- 1. System Setup (Ising Model) ---
def get_hamiltonian(L, J, h):
    dim = 2**L
    H = np.zeros((dim, dim))
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    def get_op(op_type, site):
        op = np.eye(1)
        for j in range(L):
            op = np.kron(op, op_type if j == site else I)
        return op

    for i in range(L - 1):
        H -= J * np.dot(get_op(Z, i), get_op(Z, i+1))
    for i in range(L):
        H -= h * get_op(X, i)
    return H

# Parameters
L = 4
J = 1.0
h = 1.5
H_mat = get_hamiltonian(L, J, h)
eigvals, eigvecs = eigh(H_mat)

psi0 = np.zeros(2**L); psi0[0] = 1.0
overlaps = np.abs(np.dot(eigvecs.T, psi0))**2

# --- 2. Common Rodeo Filter Function ---
def simulate_rodeo(E_scan, N, t_rms, noise_level=0.0):
    np.random.seed(42)
    signal = np.zeros_like(E_scan)
    n_realizations = 20
    
    for _ in range(n_realizations):
        ts = np.random.normal(0, t_rms, N)
        for j, Ej in enumerate(eigvals):
            filt = np.ones_like(E_scan)
            for t in ts:
                # Add depolarising noise effectively to the visibility
                visibility = 1.0 - noise_level
                filt *= visibility * np.cos((Ej - E_scan) * t / 2)**2 + (1 - visibility)/2.0
            signal += overlaps[j] * filt

    signal /= n_realizations
    
    # Introduce shot noise
    if noise_level > 0:
        shots = 1000
        signal = np.random.poisson(signal * shots) / shots
        
    return signal


# --- 4. Figure 4: Noiseless 200 points ---
E_scan_200 = np.linspace(-8, 8, 200)
signal_200 = simulate_rodeo(E_scan_200, N=10, t_rms=5.0)

# ---------------------------------------------------------------------------
#  Single-ancilla Rodeo circuit 
# ---------------------------------------------------------------------------
def create_rodeo_single_ancilla(t_list, E_target, J, h, num_qubits,
                                 trotter_steps=20):
    """
    Single-ancilla Rodeo circuit with mid-circuit measurement and
    conditional reset.  Reduces qubit count for hardware compatibility.
    """
    cycles = len(t_list)
    qr_sys = QuantumRegister(num_qubits, "sys")
    qr_anc = QuantumRegister(1, "anc")
    cr_anc = ClassicalRegister(cycles, "meas")
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
        qc.measure(qr_anc, cr_anc[m])

        # Always reset: return to |0> for any output and then applies x-gate
        qc.reset(qr_anc)
        qc.x(qr_anc) # Turns the state to |1> always

    return qc



# ---------------------------------------------------------------------------
#  Simulation parameters
# ---------------------------------------------------------------------------
J_COUPLING   = 1.0
H_FIELD      = 1.5
NUM_QUBITS   = 4
CYCLES       = 5 
SHOTS        = 2000
T_RMS        = 1.0
SCAN_POINTS  = 100
E_MIN, E_MAX = -8, 8
TROTTER      = 3

np.random.seed(42)
t_list = np.random.normal(0, T_RMS, CYCLES)
E_scan = np.linspace(E_MIN, E_MAX, SCAN_POINTS)

# ---------------------------------------------------------------------------
#  Model for the connexion with the IBM real hardware
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def run_ibm_model():
    """
    Builds circuits locally and submits TWO jobs to the IBM Cloud:
    1. Raw (No error suppression)
    2. Mitigated (With DD and Pauli Twirling)
    """
    # 1. Connect to the real machine
    real_backend = service.least_busy(operational=True, simulator=False, min_num_qubits=5)
    pm = generate_preset_pass_manager(backend=real_backend, optimization_level=1)

    # 2. Build and transpile ALL circuits locally
    isa_circuits = []
    for E in tqdm(E_scan, desc="Building circuits locally"):
        qc = create_rodeo_single_ancilla(t_list, E, J_COUPLING, H_FIELD, NUM_QUBITS, TROTTER)
        transpiled_qc = pm.run(qc)
        isa_circuits.append(transpiled_qc)

    print(f"\n--- Submitting Jobs to {real_backend.name} ---")

    # ---------------------------------------------------------
    # 3A. Submit JOB 1: RAW (Unmitigated)
    # ---------------------------------------------------------
    sampler_raw = Sampler(mode=real_backend)
    sampler_raw.options.dynamical_decoupling.enable = False
    sampler_raw.options.twirling.enable_gates = False
    sampler_raw.options.twirling.enable_measure = False
    
    print(f"Sending RAW job (shots={SHOTS})...")
    job_raw = sampler_raw.run(isa_circuits, shots=SHOTS)
    print(f"RAW Job ID: {job_raw.job_id()}")

    # ---------------------------------------------------------
    # 3B. Submit JOB 2: MITIGATED (DD + Twirling)
    # ---------------------------------------------------------
    sampler_mit = Sampler(mode=real_backend)
    sampler_mit.options.dynamical_decoupling.enable = True
    sampler_mit.options.twirling.enable_gates = True
    sampler_mit.options.twirling.enable_measure = False
    
    print(f"Sending MITIGATED job (shots={SHOTS})...")
    job_mit = sampler_mit.run(isa_circuits, shots=SHOTS)
    print(f"MITIGATED Job ID: {job_mit.job_id()}")

    # ---------------------------------------------------------
    # 4. Wait for both jobs to finish and extract data
    # ---------------------------------------------------------
    print("\nWaiting for physical execution on IBM Cloud...")
    print("If this takes too long, you can use the Job IDs to fetch data later!")
    
    result_raw = job_raw.result()
    result_mit = job_mit.result()
    print("Both datasets received successfully!\n")

    # Process the probabilities
    probs_raw = []
    probs_mit = []
    success_str = "1" * CYCLES
    
    for i in range(len(E_scan)):
        # Extract RAW counts
        counts_raw = result_raw[i].data.meas.get_counts()
        probs_raw.append(counts_raw.get(success_str, 0) / SHOTS)
        
        # Extract MITIGATED counts
        counts_mit = result_mit[i].data.meas.get_counts()
        probs_mit.append(counts_mit.get(success_str, 0) / SHOTS)

    # Return BOTH lists!
    return probs_raw, probs_mit


# --------------------------------------------------------------------------------------------------------
#  Main - The images cannot be show at the same time, it's necessary to chose raw or mitigation
#  With you have any problem, case of errors or disconnexion, you can use the code rescue.py to recovers the data.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Unpack the two lists returned by the function
    prob_raw, prob_mit = run_ibm_model()

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot RAW data (Red, dotted line)
    ax.plot(E_scan, prob_raw, "r--o", markersize=4, linewidth=1.2, label="Raw (No Suppression)")
    
    # Plot MITIGATED data (Blue, solid line)
    ax.plot(E_scan, prob_mit, "b-o", markersize=4, linewidth=2, label="Mitigated (DD + Twirling)")
    
    # Plot exact eigenvalues (Black stems) - Assuming 'eigvals' and 'overlaps' are defined earlier in your script
    if 'eigvals' in locals() and 'overlaps' in locals():
        # Using prob_mit for scaling the stems to look nice
        scale_factor = np.max(prob_mit) / np.max(overlaps) if np.max(overlaps) > 0 else 1
        ax.stem(eigvals, overlaps * scale_factor, linefmt='k:', markerfmt='ko', basefmt=' ', label='Exact')

    ax.set_title(f"IBM Quantum Hardware: Rodeo Spectrum (L={NUM_QUBITS})", fontsize=13)
    ax.set_xlabel(r"$E_{\mathrm{target}}$")
    ax.set_ylabel(r"$P(N)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    #plt.savefig("spectrum_hardware_comparison.png", dpi=200)
    plt.show()
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

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

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

#plt.figure(figsize=(10, 6))
#plt.plot(E_scan_200, signal_200, 'b-o', markersize=3, linewidth=1.2, label='Noiseless (200 pts)')
#plt.stem(eigvals, overlaps * (np.max(signal_200)/np.max(overlaps)), linefmt='k:', markerfmt='ko', basefmt=' ', label='Exact')
#plt.xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
#plt.ylabel(r"Success probability $P(N)$", fontsize=13)
#plt.title(f"Simulated Spectrum — Ising Model $L={L}$ (noiseless, 200 pts)", fontsize=14)
#plt.grid(True, alpha=0.3)
#plt.legend()
#plt.tight_layout()
#plt.show()

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
                                trotter_steps=20):
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
        #qc.p(-E_target * t, qr_anc)
        #qc.x(qr_anc)

    return qc

# Mas e se eu colocar essa função de reset sempre, ai ela vai ser aplicado para em |0> e os |1> ai é só aplicar o x
# To testando fazer isso


# ---------------------------------------------------------------------------
#  Simulation parameters
# ---------------------------------------------------------------------------
J_COUPLING   = 1.0
H_FIELD      = 1.5
NUM_QUBITS   = 4
CYCLES       = 4 
SHOTS        = 1000
T_RMS        = 5.0
SCAN_POINTS  = 50
E_MIN, E_MAX = -8, 8
TROTTER      = 20

np.random.seed(42)
t_list = np.random.normal(0, T_RMS, CYCLES)
E_scan = np.linspace(E_MIN, E_MAX, SCAN_POINTS)


# ---------------------------------------------------------------------------
#  (A)  Depolarising noise model # Com o noisy, 10 ciclos e 5000 shots ele vai de 4 min para 20 min
#  Mas com mais de uma ancila estava levando horas (1h30 - 2h30), se ficar bom com 1 ancila ai ta valendo ja.
# ---------------------------------------------------------------------------
def run_depolarising_scan():
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.0001, 1), ["x", "h", "p"]) #0.001 original
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.0001,  2), ["cx", "crx", "crz"]) #0.01 original

    # Teste com 10 ciclos ta fazendo o sinal morrer com 0,001 e 0,01 devido a profundidade do circuito
    # Teste com o codigo acima reduziu muito o sinal em 10 ciclos, Vou modficiar para erros só nos qbit de entrada
    # Demora mais ou menos 30-40 min pra rodar com o codigo acima
    # Vou testar depois com 5 ciclos pra ver se ainda tem output
    

    backend = AerSimulator(noise_model=noise_model)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    probs = []
    success_str = "1" * CYCLES
    for E in tqdm(E_scan, desc="Depolarising scan"):
        qc = create_rodeo_single_ancilla(
            t_list, E, J_COUPLING, H_FIELD, NUM_QUBITS, TROTTER)
        counts = backend.run(pm.run(qc), shots=SHOTS).result().get_counts()
        probs.append(counts.get(success_str, 0) / SHOTS)
    return probs
# Nessa parte aqui de cima o qc roda o create_rodeo para uma energia em um numero N de ciclos.
# Mas como eu uso a mesma ancilla, ela n guarda a informação após ser resetada, ent eu preciso fazer uma lista/string
# dentro da função e retorna-la junto a sendo a succes_str
# Só essa linha q eu n to me lembrando o q faz counts = backend.run(pm.run(qc), shots=SHOTS).result().get_counts()
# O qc ta retornando só o ultimo ciclo, ent mesmo q ele rode para 2000 shots, vão ser 2000 shots em 1 ciclo. 
# Preciso ver como faço para editar isso e guardar os valores dos ciclos anteriores ao ultimo e 
# junta-los nessa variavel counts 
# Aumentei o numero de registradores classicos, vamos ver se da certo.



# ---------------------------------------------------------------------------
#  (B)  IBM Manila realistic noise model
# Não ta dando sinal acima de 5 ciclos. To tentando corrigir isso
# É a profundidade do circuito, n vai dar pra ficar melhor sem tecnicas de correção
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
    success_str = "1" * CYCLES
    for E in tqdm(E_scan, desc="IBM Manila scan"):
        qc = create_rodeo_single_ancilla(
            t_list, E, J_COUPLING, H_FIELD, NUM_QUBITS, TROTTER)
        counts = backend.run(pm.run(qc), shots=SHOTS).result().get_counts()
        probs.append(counts.get(success_str, 0) / SHOTS)
    return probs


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prob_depol = run_depolarising_scan()
    #prob_manila = run_manila_scan()

    n_panels = 1 # if prob_manila is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(E_scan, prob_depol, "r-o", markersize=3, linewidth=1.2)
    axes[0].set_title("Depolarising noise", fontsize=13)
    axes[0].set_xlabel(r"$E_{\mathrm{target}}$")
    axes[0].set_ylabel(r"$P(N)$")
    axes[0].grid(True, alpha=0.3)

    #if prob_manila is not None:
    #    axes[0].plot(E_scan, prob_manila, "g-o", markersize=3, linewidth=1.2)
    #    axes[0].set_title("IBM Manila noise", fontsize=13)
    #    axes[0].set_xlabel(r"$E_{\mathrm{target}}$")
    #    axes[0].set_ylabel(r"$P(N)$")
    #    axes[0].grid(True, alpha=0.3)

    fig.suptitle(f"Noisy Rodeo Spectrum — Ising $L={NUM_QUBITS}$", fontsize=14)
    plt.tight_layout()
    plt.stem(eigvals, overlaps * (np.max(signal_200)/np.max(overlaps)), linefmt='k:', markerfmt='ko', basefmt=' ', label='Exact')
    #plt.savefig("spectrum_noisy_comparison.png", dpi=200)
    plt.show()
    #print("Saved: spectrum_noisy_comparison.png")


# AGORA É TENTAR CORRIGIR O ERRO OU TENTAR SÓ COLOCAR O ERRO NA ANCILLA INICIAL. Vou tentar a correção e depois tento implementar isso
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# --- 1. System Setup (Heisenberg XXX Model) ---
def get_heisenberg_hamiltonian(L, J=1.0):
    """Builds H = J * sum(X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})"""
    dim = 2**L
    H = np.zeros((dim, dim), dtype=complex)

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def get_op(op_type, site):
        op = np.eye(1, dtype=complex)
        for j in range(L):
            op = np.kron(op, op_type if j == site else I)
        return op

    # Interaction terms (Open Boundary)
    for i in range(L - 1):
        H += J * np.dot(get_op(X, i), get_op(X, i+1))
        H += J * np.dot(get_op(Y, i), get_op(Y, i+1))
        H += J * np.dot(get_op(Z, i), get_op(Z, i+1))

    return np.real(H) # Heisenberg XXX eigenvalues are real

# Parameters
L = 3    # Small chain
J = 1.0  # Anti-ferromagnetic coupling
H_mat = get_heisenberg_hamiltonian(L, J)
eigvals, _ = eigh(H_mat)

# Since we just want the theoretical filter map, we'll assume uniform overlap
unique_eigvals = np.unique(np.round(eigvals, 5))

def simulate_rodeo(E_scan, N, t_rms):
    np.random.seed(42)
    signal = np.zeros_like(E_scan)
    n_realizations = 20
    
    for _ in range(n_realizations):
        ts = np.random.normal(0, t_rms, N)
        for j, Ej in enumerate(unique_eigvals):
            filt = np.ones_like(E_scan)
            for t in ts:
                filt *= np.cos((Ej - E_scan) * t / 2)**2
            signal += filt / len(unique_eigvals)
    
    signal /= n_realizations
    return signal

E_scan = np.linspace(np.min(unique_eigvals)-2, np.max(unique_eigvals)+2, 300)
signal = simulate_rodeo(E_scan, N=10, t_rms=5.0)

plt.figure(figsize=(10, 6))
plt.plot(E_scan, signal, 'g-o', markersize=3, linewidth=1.2, label='Rodeo Signal')

# Scale the stems
scale = np.max(signal)
plt.stem(unique_eigvals, np.ones_like(unique_eigvals)*scale, linefmt='k:', markerfmt='ko', basefmt=' ', label='Exact Eigenvalues')

plt.xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
plt.ylabel(r"Success probability $P(N)$", fontsize=13)
plt.title(f"Simulated Spectrum — Heisenberg XXX Chain $L={L}$", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("spectrum_heisenberg.png", dpi=200)

print("Generated Heisenberg XXX plot.")

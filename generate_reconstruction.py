import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# --- 1. System Setup (Ising Model) ---
def get_hamiltonian(L, J, h):
    """Builds H = -J*sum(Z_i Z_{i+1}) - h*sum(X_i) for L spins."""
    dim = 2**L
    H = np.zeros((dim, dim))

    # Pre-compute Pauli matrices for site i
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    def get_op(op_type, site):
        op = np.eye(1)
        for j in range(L):
            op = np.kron(op, op_type if j == site else I)
        return op

    # Interaction terms (Open Boundary)
    for i in range(L - 1):
        H -= J * np.dot(get_op(Z, i), get_op(Z, i+1))

    # Transverse field terms
    for i in range(L):
        H -= h * get_op(X, i)

    return H

# Parameters
L = 4           # Number of qubits
J = 1.0         # Coupling
h = 1.5         # Field strength
dim = 2**L

# --- 2. Exact Solution (Ground Truth) ---
H_mat = get_hamiltonian(L, J, h)
eigvals, eigvecs = eigh(H_mat)

# Initial state |0000>
psi0 = np.zeros(dim); psi0[0] = 1.0
overlaps = np.abs(np.dot(eigvecs.T, psi0))**2

# --- 3. Rodeo Simulation & Plotting ---
E_scan = np.linspace(np.min(eigvals)-2, np.max(eigvals)+2, 500)
t_rms = 5.0
cycles_list = [1, 5, 10, 30] # Evolution of cycles
n_realizations = 20          # Averaging to smooth noise

fig, axes = plt.subplots(len(cycles_list), 1, figsize=(8, 10), sharex=True)

for i, N in enumerate(cycles_list):
    signal = np.zeros_like(E_scan)

    # Monte Carlo averaging
    for _ in range(n_realizations):
        ts = np.random.normal(0, t_rms, N)
        for j, Ej in enumerate(eigvals):
            filt = np.ones_like(E_scan)
            for t in ts:
                filt *= np.cos((Ej - E_scan) * t / 2)**2
            signal += overlaps[j] * filt

    signal /= n_realizations # Normalize average
    
    # Plotting
    ax = axes[i]
    ax.plot(E_scan, signal, 'b-', lw=1.5, label=f'Rodeo Signal')
    
    scale = np.max(signal) / np.max(overlaps) if np.max(overlaps) > 0 else 1
    markerline, stemlines, baseline = ax.stem(eigvals, overlaps * scale, linefmt='k:', markerfmt='ko', basefmt=' ')
    markerline.set_label('Exact Eigenvalues')

    ax.set_ylabel(f'Prob (N={N})')
    if i == 0:
        ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Spectrum Reconstruction after {N} Cycles')

axes[-1].set_xlabel(r'Target Energy $E_{\mathrm{target}}$', fontsize=12)
plt.tight_layout()
plt.savefig("reconstruction.png", dpi=200)
print("Saved reconstruction.png")

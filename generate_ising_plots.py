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

# --- 3. Figure 3: Noiseless 300 points ---
E_scan_300 = np.linspace(-8, 8, 300)
signal_300 = simulate_rodeo(E_scan_300, N=10, t_rms=5.0)

plt.figure(figsize=(10, 6))
plt.plot(E_scan_300, signal_300, 'b-o', markersize=3, linewidth=1.2, label='Noiseless (300 pts)')
plt.stem(eigvals, overlaps * (np.max(signal_300)/np.max(overlaps)), linefmt='k:', markerfmt='ko', basefmt=' ', label='Exact')
plt.xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
plt.ylabel(r"Success probability $P(N)$", fontsize=13)
plt.title(f"Simulated Spectrum — Ising Model $L={L}$ (noiseless, 300 pts)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("spectrum_noiseless_300.png", dpi=200)

# --- 4. Figure 4: Noiseless 200 points ---
E_scan_200 = np.linspace(-8, 8, 200)
signal_200 = simulate_rodeo(E_scan_200, N=10, t_rms=5.0)

plt.figure(figsize=(10, 6))
plt.plot(E_scan_200, signal_200, 'b-o', markersize=3, linewidth=1.2, label='Noiseless (200 pts)')
plt.stem(eigvals, overlaps * (np.max(signal_200)/np.max(overlaps)), linefmt='k:', markerfmt='ko', basefmt=' ', label='Exact')
plt.xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
plt.ylabel(r"Success probability $P(N)$", fontsize=13)
plt.title(f"Simulated Spectrum — Ising Model $L={L}$ (noiseless, 200 pts)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("spectrum_noiseless_200.png", dpi=200)

# --- 5. Figure 5: Noisy 50 points ---
E_scan_50 = np.linspace(-8, 8, 50)
signal_noisy = simulate_rodeo(E_scan_50, N=10, t_rms=5.0, noise_level=0.15)

plt.figure(figsize=(10, 6))
plt.plot(E_scan_50, signal_noisy, 'r-o', markersize=4, linewidth=1.2, label='Noisy (Depolarising, 50 pts)')
plt.stem(eigvals, overlaps * (np.max(signal_noisy)/np.max(overlaps)), linefmt='k:', markerfmt='ko', basefmt=' ', label='Exact')
plt.xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
plt.ylabel(r"Success probability $P(N)$", fontsize=13)
plt.title(f"Simulated Spectrum — Ising Model $L={L}$ (noisy, 50 pts)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("spectrum_noisy.png", dpi=200)

print("Generated all Ising plots internally.")

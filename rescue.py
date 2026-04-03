import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from qiskit_ibm_runtime import QiskitRuntimeService

# ==========================================
# 1. Settings to recreate the plot
# ==========================================
L = 4
J = 1.0
h = 1.5
CYCLES = 5
SHOTS = 2000
SCAN_POINTS = 100
E_scan = np.linspace(-8, 8, SCAN_POINTS)

# Recreating the matrix to plot the exact black dots
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

H_mat = get_hamiltonian(L, J, h)
eigvals, eigvecs = eigh(H_mat)
psi0 = np.zeros(2**L); psi0[0] = 1.0
overlaps = np.abs(np.dot(eigvecs.T, psi0))**2

# ==========================================
# 2. Authentication and Job Retrieval
# ==========================================
MY_API_TOKEN = "TOKEN" 
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=MY_API_TOKEN)

job_raw_id = "RAW_ID"
job_mit_id = "MIT_ID"

print("Fetching Jobs from the IBM cloud...")
job_raw = service.job(job_raw_id)
job_mit = service.job(job_mit_id)

print(f"Status RAW: {job_raw.status()}")
print(f"Status weak MITIGATED: {job_mit.status()}")
# If it's still running, it waits here. If it's done, it proceeds!
result_raw = job_raw.result()
result_mit = job_mit.result()
print("Data retrieved successfully!")

# ==========================================
# 3. Extraction of Probabilities
# ==========================================
prob_raw = []
prob_mit = []
success_str = "1" * CYCLES

for i in range(SCAN_POINTS):
    counts_raw = result_raw[i].data.meas.get_counts()
    prob_raw.append(counts_raw.get(success_str, 0) / SHOTS)
    
    counts_mit = result_mit[i].data.meas.get_counts()
    prob_mit.append(counts_mit.get(success_str, 0) / SHOTS)


# ==========================================
# 4. Plotting the Graph
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))

# Plot RAW data (Red, dashed)
ax.plot(E_scan, prob_raw, "r--o", markersize=4, linewidth=1.2, label="Raw (Without Mitigation)")

# Plot MITIGATED data (Blue, solid)
ax.plot(E_scan, prob_mit, "b-o", markersize=4, linewidth=2, label="Mitigated (DD + Twirling)")

# Plot exact eigenvalues (Black dots)
scale_factor = np.max(prob_mit) / np.max(overlaps) if np.max(overlaps) > 0 else 1
ax.stem(eigvals, overlaps * scale_factor, linefmt='k:', markerfmt='ko', basefmt=' ', label='Exact')

ax.set_title(f"IBM Quantum Hardware: Rodeo Spectrum (L={L}, Cycles={CYCLES})", fontsize=13)
ax.set_xlabel(r"$E_{\mathrm{target}}$")
ax.set_ylabel(r"$P(N)$")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def generate_dirac_spectrum(m_val, t_rms, cycles, ax, color='m'):
    # Parameters
    L = 4
    a = 1.0
    np.random.seed(123)
    t_list = np.random.normal(0, t_rms, cycles)

    # Eigenvalues of Dirac H are +/- sqrt(sin^2(k) + m^2) 
    # where sin(k) in {0, 1, 0, -1} for L=4
    # Giving roots: +/- sqrt(0 + m^2) = +/- m, and +/- sqrt(1 + m^2)
    val1 = m_val
    val2 = np.sqrt(1.0 + m_val**2)
    eigenvalues = np.array([val1, -val1, val2, -val2])
    
    def P(E_target):
        total_p = 0
        for En in eigenvalues:
            prob = 1.0
            for t in t_list:
                prob *= np.cos((En - E_target) * t / 2.0)**2
            total_p += prob
        return total_p / len(eigenvalues)

    E_scan = np.linspace(-4, 4, 300)
    probs = [P(E) for E in E_scan]

    ax.plot(E_scan, probs, color=color, linewidth=1.5, label=f"Dirac Theory (m={m_val})")
    ax.fill_between(E_scan, probs, color=color, alpha=0.2)
    
    # Stem for exact
    overlaps = np.ones_like(eigenvalues) * (np.max(probs)/1.0) # uniform overlap assumption
    ax.stem(eigenvalues, overlaps, linefmt='k:', markerfmt='ko', basefmt=' ')
    
    ax.set_ylabel(r"Success prob $P(N)$", fontsize=11)
    ax.set_title(r"Expected Dirac Spectrum ($m=" + str(m_val) + r"$)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Generate for m = 0.5
generate_dirac_spectrum(0.5, 4.0, 6, ax1, color='m')

# Generate for m = 2.0
generate_dirac_spectrum(2.0, 4.0, 6, ax2, color='c')

ax2.set_xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
plt.tight_layout()
plt.savefig("spectrum_dirac_varied.png", dpi=200)
print("Saved spectrum_dirac_varied.png")

import numpy as np
import matplotlib.pyplot as plt

def generate_dirac_spectrum():
    # Parameters matches rodeo_dirac.py
    L = 4
    m = 1.0
    a = 1.0
    cycles = 6
    t_rms = 4.0
    np.random.seed(123)
    t_list = np.random.normal(0, t_rms, cycles)

    # Momentum values for 4 sites (periodic roughly, or just use the matrix)
    # H = (1/2a) sigma_z (T - T_dag) + m sigma_x
    # For L=4, the eigenvalues of the discrete derivative (1/2i)(T - T_dag) are sin(k)
    # k = 0, pi/2, pi, 3pi/2 => sin(k) = 0, 1, 0, -1
    # Eigenvalues of Dirac H are sqrt(sin^2(k) + m^2)
    # So E are: +/- sqrt(0+1), +/- sqrt(1+1), +/- sqrt(0+1), +/- sqrt(1+1)
    # E in {1, -1, sqrt(2), -sqrt(2), 1, -1, sqrt(2), -sqrt(2)}
    eigenvalues = np.array([1.0, -1.0, np.sqrt(2), -np.sqrt(2)])
    
    # Success probability function
    def P(E_target):
        total_p = 0
        # Assume initial state has some overlap with all eigenstates for visualization
        for En in eigenvalues:
            prob = 1.0
            for t in t_list:
                prob *= np.cos((En - E_target) * t / 2.0)**2
            total_p += prob
        return total_p / len(eigenvalues) # Normalised roughly

    E_scan = np.linspace(-4, 4, 300)
    probs = [P(E) for E in E_scan]

    plt.figure(figsize=(10, 6))
    plt.plot(E_scan, probs, "m-", linewidth=1.5, label=f"Dirac Theory ({cycles} cycles)")
    plt.fill_between(E_scan, probs, color='magenta', alpha=0.2)
    plt.xlabel(r"Target energy $E_{\mathrm{target}}$", fontsize=13)
    plt.ylabel(r"Success probability $P(N)$", fontsize=13)
    plt.title(r"Expected Rodeo Spectrum — 1D Discrete Dirac  $L=4,\; m=1.0$", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("spectrum_dirac.png", dpi=200)
    print("Saved spectrum_dirac.png")

if __name__ == "__main__":
    generate_dirac_spectrum()

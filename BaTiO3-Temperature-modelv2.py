import numpy as np
import matplotlib.pyplot as plt

# Constants
a, c = 3.992, 4.036  # Lattice parameters at T = 0 K in Angstroms
beta, alpha, gamma = 1e-5, 1.0, 0.01  # Coefficients
t_pd_0, t_pp_0, t_dd_0 = 2.0, 1.0, 0.5  # Hopping parameters at T = 0 K (in eV)
E_d_0, E_p_0 = -2.0, 0.0  # On-site energies at T = 0 K (in eV)
H_DIM = 8  # Dimension of the Hamiltonian matrix

# Temperature-dependent hopping parameters
def hopping_parameter(t_0, r_0, T):
    return t_0 * np.exp(-alpha * r_0 * beta * T)

# Temperature-dependent on-site energies
def on_site_energy(E_0, T):
    return E_0 + gamma * T

# Construct the tight-binding Hamiltonian for BaTiO₃
def tight_binding_hamiltonian(k, T):
    t_pd = hopping_parameter(t_pd_0, a / 2, T)
    t_pp = hopping_parameter(t_pp_0, a / 2, T)
    t_dd = hopping_parameter(t_dd_0, a / 2, T)
    E_d = on_site_energy(E_d_0, T)
    E_p = on_site_energy(E_p_0, T)

    H = np.zeros((H_DIM, H_DIM), dtype=complex)

    # Ti 3d - O 2p interactions
    H[0, 3] = t_pd * np.exp(1j * k[0] * a / 2)
    H[1, 4] = t_pd * np.exp(1j * k[1] * a / 2)
    H[2, 5] = t_pd * np.exp(1j * k[2] * c / 2)

    # O 2p - O 2p interactions
    H[3, 4] = t_pp * np.exp(1j * (k[0] - k[1]) * a / 2)
    H[4, 5] = t_pp * np.exp(1j * (k[1] - k[2]) * a / 2)

    # Ti 3d - Ti 3d interactions
    H[0, 1] = t_dd * np.exp(1j * (k[0] - k[1]) * a / 2)
    H[1, 2] = t_dd * np.exp(1j * (k[1] - k[2]) * a / 2)

    np.fill_diagonal(H[:3], E_d)
    np.fill_diagonal(H[3:], E_p)

    H = H + H.conj().T
    return H

# High-symmetry k-points in the Brillouin zone
k_points = [
    [0, 0, 0], [np.pi / a, 0, 0], [np.pi / a, np.pi / a, 0], 
    [0, 0, np.pi / c], [0, 0, 0]
]
k_labels = ['Γ', 'X', 'M', 'Z', 'Γ']
num_points = 100

def generate_k_path(k_points, num_points):
    k_path = []
    for i in range(len(k_points) - 1):
        start, end = np.array(k_points[i]), np.array(k_points[i + 1])
        k_path.extend(start + t * (end - start) for t in np.linspace(0, 1, num_points))
    return k_path

def plot_band_structure(k_path, temperatures):
    plt.figure(figsize=(12, 8))
    for T in temperatures:
        eigenvalues = [np.linalg.eigvalsh(tight_binding_hamiltonian(k, T)) for k in k_path]
        eigenvalues = np.array(eigenvalues)
        for band in range(eigenvalues.shape[1]):
            plt.plot(range(len(k_path)), eigenvalues[:, band], label=f'T = {T} K' if band == 0 else "")
    
    plt.xticks([i * num_points for i in range(len(k_labels))], k_labels, fontsize=12)
    plt.xlabel('k-path', fontsize=14)
    plt.ylabel('Energy (eV)', fontsize=14)
    plt.title('Temperature-Dependent Band Structure of BaTiO₃', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()

k_path = generate_k_path(k_points, num_points)
temperatures = [0, 100, 300, 500]
plot_band_structure(k_path, temperatures)

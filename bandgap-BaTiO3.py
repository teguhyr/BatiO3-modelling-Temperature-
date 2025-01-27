import numpy as np
import matplotlib.pyplot as plt

# Constants
a = 3.992  # Lattice parameter at T = 0 K in Angstroms
c = 4.036  # Lattice parameter at T = 0 K in Angstroms
beta = 1e-5  # Thermal expansion coefficient (1/K)
alpha = 1.0  # Hopping parameter decay constant
gamma = 0.01  # On-site energy temperature coefficient (eV/K)

# Hopping parameters at T = 0 K (in eV)
t_pd_0 = 2.0  # Ti 3d - O 2p hopping
t_pp_0 = 1.0  # O 2p - O 2p hopping
t_dd_0 = 0.5  # Ti 3d - Ti 3d hopping

# On-site energies at T = 0 K (in eV)
E_d_0 = -2.0  # Ti 3d orbital energy
E_p_0 = 0.0   # O 2p orbital energy

# Temperature-dependent hopping parameters
def hopping_parameter(t_0, r_0, T):
    return t_0 * np.exp(-alpha * r_0 * beta * T)

# Temperature-dependent on-site energies
def on_site_energy(E_0, T):
    return E_0 + gamma * T

# Tight-binding Hamiltonian with temperature dependence
def tight_binding_hamiltonian(k, T):
    """
    Construct the tight-binding Hamiltonian for BaTiO₃ at a given k-point and temperature.
    """
    # Update hopping parameters and on-site energies with temperature
    t_pd = hopping_parameter(t_pd_0, a / 2, T)
    t_pp = hopping_parameter(t_pp_0, a / 2, T)
    t_dd = hopping_parameter(t_dd_0, a / 2, T)
    E_d = on_site_energy(E_d_0, T)
    E_p = on_site_energy(E_p_0, T)

    # Initialize Hamiltonian matrix
    H = np.zeros((8, 8), dtype=complex)  # 3 Ti 3d orbitals + 5 O 2p orbitals

    # Ti 3d - O 2p interactions
    H[0, 3] = t_pd * np.exp(1j * k[0] * a / 2)  # Ti d_z^2 - O p_z
    H[1, 4] = t_pd * np.exp(1j * k[1] * a / 2)  # Ti d_xz - O p_x
    H[2, 5] = t_pd * np.exp(1j * k[2] * c / 2)  # Ti d_yz - O p_y

    # O 2p - O 2p interactions
    H[3, 4] = t_pp * np.exp(1j * (k[0] - k[1]) * a / 2)  # O p_z - O p_x
    H[4, 5] = t_pp * np.exp(1j * (k[1] - k[2]) * a / 2)  # O p_x - O p_y

    # Ti 3d - Ti 3d interactions
    H[0, 1] = t_dd * np.exp(1j * (k[0] - k[1]) * a / 2)  # Ti d_z^2 - Ti d_xz
    H[1, 2] = t_dd * np.exp(1j * (k[1] - k[2]) * a / 2)  # Ti d_xz - Ti d_yz

    # On-site energies
    np.fill_diagonal(H[:3], E_d)  # Ti 3d orbitals
    np.fill_diagonal(H[3:], E_p)  # O 2p orbitals

    # Ensure Hermiticity
    H = H + H.conj().T
    return H
# High-symmetry k-points in the Brillouin zone
k_points = [
    [0, 0, 0],          # Gamma point
    [np.pi / a, 0, 0],  # X point
    [np.pi / a, np.pi / a, 0],  # M point
    [0, 0, np.pi / c],  # Z point
    [0, 0, 0]           # Back to Gamma
]

# Labels for high-symmetry points
k_labels = ['Γ', 'X', 'M', 'Z', 'Γ']

# Generate k-path
num_points = 100
k_path = []
for i in range(len(k_points) - 1):
    start = np.array(k_points[i])
    end = np.array(k_points[i + 1])
    k_path.extend([start + t * (end - start) for t in np.linspace(0, 1, num_points)])

# Temperatures to simulate (in K)
temperatures = [0, 100, 300, 500]

# Plot band structures at different temperatures
plt.figure(figsize=(10, 6))
for T in temperatures:
    eigenvalues = []
    for k in k_path:
        H_k = tight_binding_hamiltonian(k, T)
        eigvals = np.linalg.eigvalsh(H_k)
        eigenvalues.append(eigvals)
    eigenvalues = np.array(eigenvalues)
    for band in range(eigenvalues.shape[1]):
        plt.plot(range(len(k_path)), eigenvalues[:, band], label=f'T = {T} K' if band == 0 else "")

# Add high-symmetry point labels
plt.xticks(
    [i * num_points for i in range(len(k_labels))],
    k_labels,
    fontsize=12
)
plt.xlabel('k-path', fontsize=14)
plt.ylabel('Energy (eV)', fontsize=14)
plt.title('Temperature-Dependent Band Structure of BaTiO₃', fontsize=16)
plt.grid(True)
plt.legend()
plt.show()

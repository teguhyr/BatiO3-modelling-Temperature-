import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Constants
a = 4.00  # Lattice parameter a at room temperature
c = 4.00  # Lattice parameter c at room temperature
beta = 1e-5  # Thermal expansion coefficient

# Function to update lattice parameters with temperature
def lattice_parameters(T):
    a_T = a * (1 + beta * T)  # Temperature-dependent lattice parameter a
    c_T = c * (1 + beta * T)  # Temperature-dependent lattice parameter c
    return a_T, c_T

# Function to plot the 3D unit cell
def plot_unit_cell_3d(T):
    a_T, c_T = lattice_parameters(T)
    
    # Atomic positions in fractional coordinates
    positions_frac = np.array([
        [0, 0, 0],  # Ba
        [0.5, 0.5, 0.5],  # Ti
        [0.5, 0.5, 0],  # O1
        [0.5, 0, 0.5],  # O2
        [0, 0.5, 0.5],  # O3
    ])
    
    # Convert fractional coordinates to Cartesian coordinates
    positions_cart = np.array([[x * a_T, y * a_T, z * c_T] for x, y, z in positions_frac])
    
    # Define the unit cell edges
    edges = [
        [[0, 0, 0], [a_T, 0, 0]],
        [[0, 0, 0], [0, a_T, 0]],
        [[0, 0, 0], [0, 0, c_T]],
        [[a_T, 0, 0], [a_T, a_T, 0]],
        [[a_T, 0, 0], [a_T, 0, c_T]],
        [[0, a_T, 0], [a_T, a_T, 0]],
        [[0, a_T, 0], [0, a_T, c_T]],
        [[0, 0, c_T], [a_T, 0, c_T]],
        [[0, 0, c_T], [0, a_T, c_T]],
        [[a_T, a_T, 0], [a_T, a_T, c_T]],
        [[a_T, 0, c_T], [a_T, a_T, c_T]],
        [[0, a_T, c_T], [a_T, a_T, c_T]],
    ]
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    ax.scatter(positions_cart[:, 0], positions_cart[:, 1], positions_cart[:, 2], color='r', s=100)
    
    # Plot edges
    for edge in edges:
        edge = np.array(edge)
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    T = 300  # Temperature in Kelvin
    plot_unit_cell_3d(T)

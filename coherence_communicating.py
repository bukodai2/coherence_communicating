import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy

def generate_bipartite_mixed_state():
    # Define dimensions for the bipartite system (2 qubits => 4x4 system)
    dim_a = 2  # Dimension of subsystem A
    dim_b = 2  # Dimension of subsystem B
    total_dim = dim_a * dim_b  # Total system dimension

    # Step 1: Generate random pure states for the bipartite system
    num_pure_states = 100  # Number of pure states to mix
    weights = np.random.rand(num_pure_states)
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    pure_states = []
    density_matrices = []

    for _ in range(num_pure_states):
        psi = np.random.randn(total_dim) + 1j * np.random.randn(total_dim)  # Complex wavefunction
        psi /= np.linalg.norm(psi)  # Normalize the wavefunction
        pure_states.append(psi)
        density_matrices.append(np.outer(psi, psi.conj()))  # Compute density matrix for this pure state

    # Step 2: Combine density matrices to form a mixed state
    mixed_density_matrix = sum(w * dm for w, dm in zip(weights, density_matrices))

    # Step 3: Extract subsystems' wavefunctions (A and B)
    subsystem_a_wavefunctions = []
    subsystem_b_wavefunctions = []
    for psi in pure_states:
        psi_reshaped = psi.reshape(dim_a, dim_b)
        u, s, vh = np.linalg.svd(psi_reshaped)  # Singular Value Decomposition
        subsystem_a_wavefunctions.append(u[:, 0])  # Take the first column as dominant A state
        subsystem_b_wavefunctions.append(vh[0, :])  # Take the first row as dominant B state

    return pure_states, mixed_density_matrix, subsystem_a_wavefunctions, subsystem_b_wavefunctions

def calculate_quantum_metrics(density_matrix):
    # 1. Sum of absolute values of all off-diagonal elements (L1 norm)
    off_diagonal_elements = density_matrix - np.diag(np.diag(density_matrix))  # Zero out diagonal elements
    l1_norm = np.sum(np.abs(off_diagonal_elements))

    # 2. Root of the sum of squared absolute values of all off-diagonal elements (L2 norm)
    l2_norm = np.sqrt(np.sum(np.abs(off_diagonal_elements)**2))

    # 3. Fidelity-related metric: 1 - Fidelity with diagonal density matrix
    sigma = np.diag(np.diag(density_matrix))  # Diagonal matrix with the same diagonal elements
    sqrt_sigma = sqrtm(sigma)
    sqrt_sigma_rho_sqrt_sigma = sqrtm(sqrt_sigma @ density_matrix @ sqrt_sigma)
    fidelity = np.real(np.trace(sqrt_sigma_rho_sqrt_sigma))**2
    fidelity_metric = 1 - fidelity

    # 4. Von Neumann entropy difference (sigma entropy - density matrix entropy)
    eigenvalues_sigma = np.diag(sigma).real  # Ensure real values
    eigenvalues_sigma = np.clip(eigenvalues_sigma, 0, None)  # Clip to non-negative
    if np.sum(eigenvalues_sigma) > 0:
        eigenvalues_sigma /= np.sum(eigenvalues_sigma)  # Normalize to form a probability distribution
    sigma_entropy = entropy(eigenvalues_sigma, base=2)  # Base-2 entropy

    eigenvalues_rho = np.linalg.eigvalsh(density_matrix).real  # Compute eigenvalues of rho
    eigenvalues_rho = np.clip(eigenvalues_rho, 0, None)  # Clip to non-negative
    if np.sum(eigenvalues_rho) > 0:
        eigenvalues_rho /= np.sum(eigenvalues_rho)  # Normalize to form a probability distribution
    rho_entropy = entropy(eigenvalues_rho, base=2)  # Base-2 entropy

    von_neumann_entropy_difference = sigma_entropy - rho_entropy

    # Additional metrics based on L2 norm (D)
    d_squared = l2_norm**2
    metric_1 = (1 / 4) * d_squared
    metric_2 = -np.log(1 - d_squared)

    return l1_norm, l2_norm, fidelity_metric, von_neumann_entropy_difference, metric_1, metric_2

# Run the function
pure_states, mixed_density_matrix, subsystem_a_wavefunctions, subsystem_b_wavefunctions = generate_bipartite_mixed_state()

# Calculate quantum metrics
l1_norm, l2_norm, fidelity_metric, von_neumann_entropy_difference, metric_1, metric_2 = calculate_quantum_metrics(mixed_density_matrix)

# Display the results
print("Pure states (total wavefunctions):")
for i, psi in enumerate(pure_states):
    print(f"State {i + 1}: {psi}\n")

print("Mixed density matrix (rho):")
print(mixed_density_matrix, "\n")

print("Subsystem A wavefunctions:")
for i, psi_a in enumerate(subsystem_a_wavefunctions):
    print(f"Wavefunction A for state {i + 1}: {psi_a}\n")

print("Subsystem B wavefunctions:")
for i, psi_b in enumerate(subsystem_b_wavefunctions):
    print(f"Wavefunction B for state {i + 1}: {psi_b}\n")


print("Quantum metrics:")
print(f"1. Sum of absolute values of all off-diagonal elements (L1 norm): {l1_norm}")
print(f"2. Root of the sum of squared absolute values of all off-diagonal elements (L2 norm): {l2_norm}")
print(f"3. 1 - Fidelity with diagonal density matrix: {fidelity_metric}")
print(f"4. Metric 1 (1/4 * D^2): {metric_1}")
print(f"5. Von Neumann entropy difference (sigma - rho): {von_neumann_entropy_difference}")
print(f"6. Metric 2 (-log(1 - D^2)): {metric_2}")

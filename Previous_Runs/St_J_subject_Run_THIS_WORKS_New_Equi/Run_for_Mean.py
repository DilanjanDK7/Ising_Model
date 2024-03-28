import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import dual_annealing
from nilearn.connectome import ConnectivityMeasure
import seaborn as sns

def normalize_matrix(matrix):
    max_val = np.max(matrix)
    return matrix / max_val if max_val != 0 else matrix

def calculate_simulated_fc(time_series, kind='correlation'):
    connectivity_measure = ConnectivityMeasure(kind=kind)
    correlation_matrix = connectivity_measure.fit_transform([time_series])[0]
    return correlation_matrix

def metropolis_step_all_spins(spin_array, beta, Jij=None, mu=None, alpha=1.0):
    N = spin_array.shape[0]
    for i in np.random.permutation(N):
        local_T = (mu[i] ** alpha) / beta if mu is not None else 1 / beta
        local_beta = 1 / local_T
        delta_E = 2 * spin_array[i] * np.sum(Jij[i] * spin_array)
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E * local_beta):
            spin_array[i] *= -1
    return spin_array

def simulate_spin_system(N, steps, Jij, mu, beta, alpha=1.0):
    spin_array = np.random.choice([-1, 1], size=(N,))
    for _ in range(steps):
        spin_array = metropolis_step_all_spins(spin_array, beta, Jij, mu, alpha)
    return spin_array

def discrepancy_function(params, empirical_fc, N, steps, Jij=None, mu=None):
    temperature, alpha = params
    beta = 1.0 / temperature
    simulated_time_series = simulate_spin_system(N, steps, Jij, mu, beta, alpha)
    simulated_fc = calculate_simulated_fc(np.reshape(simulated_time_series, (1, -1)))
    discrepancy = np.linalg.norm(empirical_fc - simulated_fc, 'fro')
    return discrepancy

def optimize_parameters(empirical_fc, N, steps, Jij, mu, bounds=((0.01, 10), (-3, 3))):
    result = dual_annealing(
        discrepancy_function,
        bounds,
        args=(empirical_fc_no_diag, N, steps_eq, steps_mc, Jij, mu),
        maxiter=500,  # Increase the maximum number of iterations
        maxfun=500,  # Increase the maximum number of function evaluations
        initial_temp=3,  # Adjust the initial temperature if needed
        restart_temp_ratio=2e-3,  # Adjust the restart temperature ratio
        visit=1.5,  # Increase the visit parameter
        accept=-100.0  # Adjust the acceptance parameter
    )
    return result.x

def process_files(empirical_fc_path, jij_path, subject_id, parcellation, output_folder):
    empirical_fc = np.loadtxt(empirical_fc_path, delimiter=',')
    Jij = np.loadtxt(jij_path, delimiter=',')
    Jij = normalize_matrix(Jij)
    mu = np.random.rand(Jij.shape[0])  # Example: Random mu values, replace with actual data if available
    N = empirical_fc.shape[0]
    steps = 1000  # Example: Number of Metropolis steps, adjust as needed

    optimized_params = optimize_parameters(empirical_fc, N, steps, Jij, mu)
    print(f"Optimized parameters for subject {subject_id}, parcellation {parcellation}: {optimized_params}")
    return optimized_params

# Example Usage
subject_id = "subject1"
parcellation = "exampleParcellation"
output_folder = "output"
empirical_fc_path = "path_to_empirical_fc.csv"  # Update this path
jij_path = "path_to_jij.csv"  # Update this path
optimized_params = process_files(empirical_fc_path, jij_path, subject_id, parcellation, output_folder)

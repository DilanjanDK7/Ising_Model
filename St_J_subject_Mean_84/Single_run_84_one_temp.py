import numpy as np

from Full_Core import *
import os
import matplotlib.pyplot as plt
import pandas as pd

def run_discrepancy_simulation_fixed_temp(N_runs, temperature, empirical_fc, N, steps_eq, steps_mc, Jij, mu, output_folder, global_results=None,global_results_2=None):
    """
    Run the discrepancy function multiple times for a fixed temperature, collect simulation results,
    and save plots and time series data for each run.

    Parameters:
    - N_runs: Number of times to run the discrepancy simulation.
    - temperature: Fixed temperature for all simulations.
    - empirical_fc: Empirical functional connectivity matrix.
    - N: Number of nodes in the network.
    - steps_eq: Number of steps for equilibration in the simulation.
    - steps_mc: Number of Monte Carlo steps in the simulation.
    - Jij: Interaction matrix for the simulation.
    - mu: Mean field value for the simulation.
    - output_folder: Folder to save the output plots and time series data.
    - global_results: A dictionary to store the results of each run.

    Returns:
    - A dictionary containing the results of each simulation run.
    """
    if global_results is None:
        global_results = {'temperature': [], 'magnetization': [], 'susceptibility': [], 'specific_heat': [], 'time_series': [], 'energy': [], 'data': []}
        global_results_2= {'temperature': [], 'magnetization': [], 'susceptibility': [], 'specific_heat': [], 'time_series': [], 'energy': [], 'data': []}

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(N_runs):
        # Use the provided fixed temperature
        alpha = 0  # Example range

        # Run the discrepancy function
        distance = discrepancy_function((temperature, alpha), empirical_fc, N, steps_eq, steps_mc, Jij, mu, global_results,global_results_2)

        print(f"Run {i+1}/{N_runs} completed with temperature: {temperature}, alpha: {alpha}, distance: {distance}")

        # Saving the simulated time series for each node in a CSV file
        time_series_df = pd.DataFrame(global_results['time_series'][-1])
        time_series_file = os.path.join(output_folder, f"time_series_run_{i+1}.csv")
        time_series_df.to_csv(time_series_file, index=False)

        x_steps = range(steps_mc)  # Create a sequence for the x-axis representing the Monte Carlo steps

        # Update the plotting sections
        plt.figure()
        plt.plot(x_steps, global_results_2['magnetization'][-steps_mc:], 'x', linestyle='none', label='Magnetization')
        # plt.plot(x_steps, global_results_2['susceptibility'][-steps_mc:], 'x', linestyle='none', label='Susceptibility')
        plt.xlabel('Monte Carlo steps')
        plt.ylabel('Value')
        plt.title(f'Magnetization and Susceptibility vs Monte Carlo Steps (Run {i + 1})')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'mag_and_suscept_vs_steps_mc_run_{i + 1}.png'))

        plt.figure()
        plt.plot(x_steps, global_results_2['energy'][-steps_mc:], 'x', linestyle='none', label='Energy')
        plt.xlabel('Monte Carlo steps')
        plt.ylabel('Energy')
        plt.title(f'Energy vs Monte Carlo Steps (Run {i + 1})')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'energy_vs_steps_mc_run_{i + 1}.png'))
    return global_results

# Example usage
N_runs = 10
fixed_temperature = 0.1
N = 84
steps_eq = 1
steps_mc = 10000
empirical_fc = np.random.rand(N, N)

Jij = np.loadtxt('/home/brainlab-qm/Desktop/New_test/To_Analyze_84/sub-Sub7/atlas_NMI_2mm.nii/Jij.csv', delimiter=',')
mu = None
output_folder = '/home/brainlab-qm/Desktop/Ising_test_29_03/Output/Testing_Random_Ising'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
global_results = run_discrepancy_simulation_fixed_temp(N_runs, fixed_temperature, empirical_fc, N, steps_eq, steps_mc, Jij, mu, output_folder)

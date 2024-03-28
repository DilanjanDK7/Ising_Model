from St_J_subject_Run.Backups.Core_Ising import *
import os
import numpy as np

# import warnings
# warnings.filterwarnings('ignore')

def calculate_matrix_correlations(empirical_fc, simulated_fc, jij):
    """
    Calculates and returns the correlations between empirical FC and simulated FC,
    empirical FC and Jij, and simulated FC and Jij.

    Parameters:
    - empirical_fc: numpy array representing the empirical functional connectivity matrix.
    - simulated_fc: numpy array representing the simulated functional connectivity matrix.
    - jij: numpy array representing the Jij structural connectivity matrix.

    Returns:
    - A dictionary containing the correlation coefficients between:
      empirical and simulated FC, empirical FC and Jij, and simulated FC and Jij.
    """

    # Flatten the matrices to simplify correlation calculations
    empirical_fc_flat = empirical_fc.flatten()
    simulated_fc_flat = simulated_fc.flatten()
    jij_flat = jij.flatten()

    # Calculate the correlations
    corr_empirical_simulated = np.corrcoef(empirical_fc_flat, simulated_fc_flat)[0, 1]
    corr_empirical_jij = np.corrcoef(empirical_fc_flat, jij_flat)[0, 1]
    corr_simulated_jij = np.corrcoef(simulated_fc_flat, jij_flat)[0, 1]

    # Return the results in a dictionary
    correlations = {
        'empirical_simulated': corr_empirical_simulated,
        'empirical_jij': corr_empirical_jij,
        'simulated_jij': corr_simulated_jij
    }

    return correlations

def calculate_matrix_correlations_and_norms(empirical_fc, simulated_fc, jij):
    """
    Calculates and returns the correlations and Frobenius norms between empirical FC and simulated FC,
    empirical FC and Jij, and simulated FC and Jij.

    Parameters:
    - empirical_fc: numpy array representing the empirical functional connectivity matrix.
    - simulated_fc: numpy array representing the simulated functional connectivity matrix.
    - jij: numpy array representing the Jij structural connectivity matrix.

    Returns:
    - A dictionary containing the correlation coefficients and Frobenius norms between:
      empirical and simulated FC, empirical FC and Jij, and simulated FC and Jij.
    """

    # Flatten the matrices to simplify correlation calculations
    empirical_fc_flat = empirical_fc.flatten()
    simulated_fc_flat = simulated_fc.flatten()
    jij_flat = jij.flatten()

    # Calculate the correlations
    corr_empirical_simulated = np.corrcoef(empirical_fc_flat, simulated_fc_flat)[0, 1]
    corr_empirical_jij = np.corrcoef(empirical_fc_flat, jij_flat)[0, 1]
    corr_simulated_jij = np.corrcoef(simulated_fc_flat, jij_flat)[0, 1]

    # Calculate Frobenius norms
    fro_norm_empirical_simulated = np.linalg.norm(empirical_fc - simulated_fc, 'fro')
    fro_norm_empirical_jij = np.linalg.norm(empirical_fc - jij, 'fro')
    fro_norm_simulated_jij = np.linalg.norm(simulated_fc - jij, 'fro')

    # Return the results in a dictionary
    results = {
        'correlations': {
            'empirical_simulated': corr_empirical_simulated,
            'empirical_jij': corr_empirical_jij,
            'simulated_jij': corr_simulated_jij
        },
        'frobenius_norms': {
            'empirical_simulated': fro_norm_empirical_simulated,
            'empirical_jij': fro_norm_empirical_jij,
            'simulated_jij': fro_norm_simulated_jij
        }
    }

    return results


def evaluate_similarity(correlation, frobenius_norm):
    """
    Evaluates the similarity based on correlation and Frobenius norm.

    Parameters:
    - correlation: Correlation coefficient between two matrices.
    - frobenius_norm: Frobenius norm of the difference between two matrices.

    Returns:
    - A string describing the evaluation of similarity.
    """
    corr_eval = "good" if abs(correlation) > 0.5 else "poor"
    fro_eval = "close" if frobenius_norm < np.median(frobenius_norm) else "distant"

    return f"Correlation is {corr_eval} and similarity is {fro_eval}"


def print_results_with_evaluation(results):
    print("Correlations between matrices:")
    for key, value in results['correlations'].items():
        print(f"{key}: {value:.4f} - {evaluate_similarity(value, results['frobenius_norms'][key])}")

    print("\nFrobenius norms between matrices:")
    fro_norm_values = list(results['frobenius_norms'].values())
    for key, value in results['frobenius_norms'].items():
        closer = "more similar" if value == min(fro_norm_values) else "less similar"
        print(f"{key}: {value:.4f} - This pair is {closer} to each other compared to others.")
def optimize_and_simulate(time_series_path, N, steps_eq, steps_mc, output_folder, Jij=None,mu=None):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: Optimization
    optimized_params = optimize_parameters(time_series_path, N, steps_eq, steps_mc, Jij, bounds=((0.01, 10), (-3, 3)),mu=mu,output_folder=output_folder)
    temp_optimized, alpha_optimized = optimized_params

    # Convert optimized temperature to beta
    beta_optimized = 1.0 / temp_optimized

    # Step 2: Simulation with Optimized Parameters
    simulation_start_time = time.time()
    _, _, _, _, time_series_optimized = simulation_task(
        (N, beta_optimized, steps_eq, steps_mc, Jij, None, alpha_optimized, True))
    simulation_end_time = time.time()

    # Step 3: Calculate Functional Connectivity
    fc_matrix_optimized = calculate_simulated_fc(time_series_optimized)

    # Step 4: Saving Results
    # Save Functional Connectivity and Time Series
    np.save(os.path.join(output_folder, 'Simulated_fc_matrix_optimized.npy'), fc_matrix_optimized)
    np.save(os.path.join(output_folder, 'time_series_optimized.npy'), time_series_optimized)

    # Plot and save plots (Optional)
    # plot_fc = plot_functional_connectivity(fc_matrix_optimized)
    plot_ts = plot_time_series(time_series_optimized)
    # plot_fc.savefig(os.path.join(output_folder, 'Simulated_fc_plot_optimized.png'))
    plot_ts.savefig(os.path.join(output_folder, 'ts_plot_optimized.png'))

    # Visualizing the correlation matrix
    plt.close()
    plt.figure(figsize=(10, 10))
    sns.heatmap(fc_matrix_optimized, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Simulated Functional Connectivity Matrix")
    plt.savefig(os.path.join(output_folder, 'Simulated_fc_plot_optimized.png'))

    if Jij is not None:
        Jij= normalize_matrix(Jij)
        plt.close()
        plt.figure(figsize=(10, 10))
        sns.heatmap(Jij, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Jij- Structural Connectivity")
        plt.savefig(os.path.join(output_folder, 'Jij.png'))
        # Calculate correlations
        empirical_fc = extract_rho(time_series_path)

        # Calculate correlations and Frobenius norms
        results = calculate_matrix_correlations_and_norms(empirical_fc, fc_matrix_optimized, Jij)

        # Print the correlation results and Frobenius norms with evaluations
        print_results_with_evaluation(results)

    # Generate and Save HTML Log
    timing_info = {
        'simulation': simulation_end_time - simulation_start_time,
        'plotting': time.time() - simulation_end_time
    }
    log_path = os.path.join(output_folder, 'simulation_log.html')
    generate_html_log([], [{'name': 'FC Matrix Optimized', 'dimension': fc_matrix_optimized.shape}], timing_info,
                      temp_optimized, [], [], output_path=log_path)


    return optimized_params, os.path.join(output_folder, 'fc_matrix_optimized.npy'), os.path.join(output_folder,
                                                                                                  'time_series_optimized.npy'), log_path


# # Ensure reproducibility
# np.random.seed(42)
np.random.seed(7)
time_series_path = "/home/brainlab-qm/Desktop/Test_Ising/Test_visual/time_series.csv"
N = 5
steps_eq = 100
steps_mc = 2500
output_main_folder = output_folder="/home/brainlab-qm/Desktop/Test_Ising/Test_visual/"
output_subfolder = "Testing_with_all"
output_path = os.path.join(output_main_folder, output_subfolder)
os.makedirs(output_path, exist_ok=True)
Jij = np.loadtxt("/home/brainlab-qm/Desktop/Test_Ising/Test_visual/Jij.csv",delimiter=',')  # Assuming Jij will be defined or loaded before calling the function
# mu = np.loadtxt("/home/brainlab-qm/Desktop/Test_Ising/Test_visual/features.txt",delimiter=',')   # Assuming mu will be defined or loaded before calling the function
mu = np. random.rand(5)
Jij= np.random.rand(5,5)
# Jij = None
# mu = None

optimize_and_simulate(time_series_path, N, steps_eq, steps_mc, output_path, Jij=Jij, mu=mu)
print(" Succesfully Completed ")

# This emperical to jij changes , find a solution to that 88 have jij constant to take readings
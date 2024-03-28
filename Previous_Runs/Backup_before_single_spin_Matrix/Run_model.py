from Core_Ising import *
import os
import numpy as np

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
    fc_matrix_optimized = calculate_empirical_fc(time_series_optimized)

    # Step 4: Saving Results
    # Save Functional Connectivity and Time Series
    np.save(os.path.join(output_folder, 'Simulated_fc_matrix_optimized.npy'), fc_matrix_optimized)
    np.save(os.path.join(output_folder, 'time_series_optimized.npy'), time_series_optimized)

    # Plot and save plots (Optional)
    plot_fc = plot_functional_connectivity(fc_matrix_optimized)
    plot_ts = plot_time_series(time_series_optimized)
    plot_fc.savefig(os.path.join(output_folder, 'Simulated_fc_plot_optimized.png'))
    plot_ts.savefig(os.path.join(output_folder, 'ts_plot_optimized.png'))

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


jij=np.loadtxt("/home/brainlab-qm/Desktop/Test_Ising/Jij.csv",delimiter=',')
mu = np.loadtxt("/home/brainlab-qm/Desktop/Test_Ising/features.txt",delimiter=',')
# Ensure reproducibility
np.random.seed(42)
optimize_and_simulate(time_series_path="/home/brainlab-qm/Desktop/Test_Ising/_image_parcellation_path_..media..ubuntu..usbdata..Pipeline..Data..Parcellation..rsn_parcellations..FrontoParietal..FrontoParietal_parcellation_5.nii/time_series.csv", N=5, steps_eq=1000, steps_mc=1000, output_folder="/home/brainlab-qm/Desktop/Test_Ising/with_mu", Jij=jij,mu=mu)
print(" Succesfully Completed ")
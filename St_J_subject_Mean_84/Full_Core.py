import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import time
from jinja2 import Template
from scipy.signal import find_peaks
from multiprocessing import Pool
from scipy.optimize import dual_annealing
from scipy.stats import pearsonr
import scipy
import seaborn as sns
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.express as px


def process_files(fmri_path, pet_path, jij_path,subject_id,parcellation,base_output_folder):
    """
    Placeholder function to process or perform calculations on the fmri, pet, and jij files.

    Parameters:

    - fmri_path: Path to the fmri file.
    - pet_path: Path to the pet file.
    - jij_path: Path to the jij file.

    This function should contain the specific analysis or processing steps for the provided files.
    For demonstration purposes, it just prints the paths of the files.
    """
    # Placeholder for actual processing logic
    output_path= create_individual_output_folders(base_output_folder, subject_id, parcellation)
    print(f"Processing:\nFMRI: {fmri_path}\nPET: {pet_path}\nJIJ: {jij_path}")
    time_series_path = fmri_path
    N = 5
    steps_eq = N*100
    steps_mc = 2000
    Jij = np.loadtxt(jij_path, delimiter=',')
    # Jij =None
    # mu=None
    mu = np.loadtxt(pet_path, delimiter=',')
    Jij = normalize_matrix(Jij) # Normalizing Jij
    optimised =optimize_and_simulate(time_series_path, N, steps_eq, steps_mc, output_path, Jij=Jij, mu=mu)
    description_array = ['Optimal_Temperature', 'Optimal_Alpha', 'Parcellation', 'Subject','correlations']
    value_array = [optimised[0][0], optimised[0][1], parcellation, subject_id,optimised[1]]
    write_values_to_file(output_path, description_array, value_array)

    print(" Succesfully Completed ")
    return optimised

# # Ensure reproducibility
# np.random.seed(42)


# def initialize_spin_matrix(N):
#     return np.random.choice([-1, 1], size=(N, N))

def generate_and_save_graphs(global_results, output_folder, mu=None):
    df_results = pd.DataFrame(global_results['data'])

    if mu is not None:
        # Dynamic 3D Graph for temperature, alpha, and distance using Plotly
        fig = px.scatter_3d(df_results, x='temperature', y='alpha', z='distance', color='distance',
                            title='3D Scatter Plot of Temperature, Alpha, and Distance',
                            labels={'temperature': 'Temperature', 'alpha': 'Alpha', 'distance': 'Distance'})
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        # Save the interactive plot as HTML
        fig.write_html(os.path.join(output_folder, '3D_Graph_Temperature_Alpha_Distance.html'))
    else:
        # 2D Graph for temperature and distance using matplotlib (remains unchanged)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 7))
        for distance_type, group in df_results.groupby('distance_type'):
            plt.scatter(group['temperature'], group['distance'], label=f'Distance Type: {distance_type}')
        plt.xlabel('Temperature')
        plt.ylabel('Distance')
        plt.title('2D Scatter Plot of Temperature and Distance')
        plt.legend()
        plt.savefig(os.path.join(output_folder, '2D_Graph_Temperature_Distance.png'))
        plt.close()
# def initialize_spin_matrix(N):
#     return np.random.choice([-1, 1], size=(N))

def initialize_spin_matrix(N):
    # Ensure an equal number of -1 and +1 spins for high energy configuration
    num_positives = N // 2
    num_negatives = N - num_positives
    spins = np.array([1] * num_positives + [-1] * num_negatives)
    np.random.shuffle(spins)
    return spins
def normalize_matrix(matrix):
    max_val = np.max(matrix)
    normalized_matrix = matrix / max_val
    return normalized_matrix
def metropolis_step_all_spins(spin_array, beta, Jij=None, mu=None, alpha=1.0, J_default=1.0):
    if Jij is not None:
        Jij = normalize_matrix(Jij)

    N = spin_array.shape[0]  # Total number of spins

    # Function to calculate delta_E for nearest neighbors if Jij is not provided
    def delta_E_nearest_neighbors(i):
        return 2 * J_default * spin_array[i] * (spin_array[(i - 1) % N] + spin_array[(i + 1) % N])

    rand_indices = np.random.permutation(N)  # Create a random permutation of indices

    for i in rand_indices:  # Iterate over spins in the order of the random permutation
        local_T = (mu[i] ** alpha) / beta if mu is not None else 1 / beta
        local_beta = 1 / local_T

        if Jij is not None:
            # Calculate the change in energy if spin i were flipped, using Jij for interactions
            delta_E = 0
            for j in range(N):  # Loop over all spins to account for interactions
                if i != j:  # Avoid self-interaction
                    delta_E += 2 * Jij[i, j] * spin_array[i] * spin_array[j]
        else:
            # Calculate the change in energy for nearest neighbors if Jij is not provided
            delta_E = delta_E_nearest_neighbors(i)

        # Decide whether to flip the spin
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E * local_beta):
            spin_array[i] *= -1

    return spin_array
def calculate_observables(spin_array, beta, Jij=None):
    N = spin_array.shape[0]
    # Calculate magnetization for 1D array
    mag = np.abs(np.sum(spin_array)) / N

    if Jij is None:
        # Calculate energy for nearest-neighbor interactions in 1D array
        energy = -np.sum(spin_array * (
                np.roll(spin_array, 1) +
                np.roll(spin_array, -1)) / 2.0)
    else:
        # Direct calculation of energy using Jij for given interactions in 1D
        energy = 0
        for i in range(N):
            for j in range(N):
                # Calculate interaction energy contribution, avoiding double-counting
                energy += -Jij[i, j] * spin_array[i] * spin_array[j]
        # No need to adjust for double-counting since each pair is considered exactly once

    return mag, energy
def simulation_task(params):
    N, beta, steps_eq, steps_mc, Jij, mu, alpha, collect_time_series,global_results_2 = params
    spin_matrix = initialize_spin_matrix(N)
    for _ in range(steps_eq):
        # metropolis_step(spin_matrix, beta, Jij, mu, alpha)
        spin_matrix = metropolis_step_all_spins(spin_matrix, beta, Jij, mu, alpha)

    mags, energies, time_series = [], [], []
    for _ in range(steps_mc):
        # metropolis_step(spin_matrix, beta, Jij)
        spin_matrix = metropolis_step_all_spins(spin_matrix, beta, Jij, mu, alpha)
        mag, energy = calculate_observables(spin_matrix, beta, Jij)
        mags.append(mag)
        energies.append(energy)
        if collect_time_series:
            time_series.append(spin_matrix.copy())
        global_results_2['magnetization'].append(mag)
        global_results_2['energy'].append(energy)

    mag_mean = np.mean(mags)
    energy_mean = np.mean(energies)
    susceptibility = (np.var(mags) * N ** 2) * beta
    specific_heat = (np.var(energies) * N ** 2) * beta ** 2

    return mag_mean, energy_mean, susceptibility, specific_heat, time_series
def run_simulation_parallel(N, temperatures, steps_eq, steps_mc, Jij=None, Tc=None, specific_temp=None):
    beta_values = 1.0 / temperatures
    pool = Pool(processes=4)  # Adjust number of processes based on your system
    tasks = [(N, beta, steps_eq, steps_mc, Jij, mu, alpha, collect_time_series) for beta in beta_values]
    results = pool.map(simulation_task, tasks)
    pool.close()
    pool.join()

    results_dict = {'magnetizations': [], 'susceptibilities': [], 'specific_heats': [], 'energies': [],
                    'time_series': {}}
    for i, (mag_mean, energy_mean, susceptibility, specific_heat, time_series) in enumerate(results):
        results_dict['magnetizations'].append(mag_mean)
        results_dict['energies'].append(energy_mean)
        results_dict['susceptibilities'].append(susceptibility)
        results_dict['specific_heats'].append(specific_heat)
        if temperatures[i] == Tc or temperatures[i] == specific_temp:
            results_dict['time_series'][temperatures[i]] = time_series

    # Estimate the critical temperature based on the peak of susceptibility if not provided
    if Tc is None:
        susceptibilities = np.array(results_dict['susceptibilities'])
        peaks, _ = find_peaks(susceptibilities)
        Tc_estimated = temperatures[peaks[0]] if len(peaks) > 0 else None
    else:
        Tc_estimated = Tc

    return results_dict, Tc_estimated
def calculate_functional_connectivity(time_series):
    # Assuming time_series is a list of N x N matrices
    # Flatten each matrix and calculate the correlation across all time points for each pair of sites
    time_series_flat = [ts.flatten() for ts in time_series]
    correlation_matrix = np.corrcoef(time_series_flat)
    return correlation_matrix
def plot_functional_connectivity(fc_matrix):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(fc_matrix, interpolation='nearest')
    fig.colorbar(cax)
    plt.title('Functional Connectivity')
    return fig
def plot_time_series(time_series):
    # Assuming time_series is a list of N x N matrices, plotting the magnetization over time for a single site as an example
    magnetizations = [np.mean(ts) for ts in time_series]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(magnetizations, label='Magnetization')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Magnetization')
    ax.set_title('Time Series of Magnetization')
    ax.legend()
    return fig

    # def run_simulation_parallel(N, temperatures, steps_eq, steps_mc, Jij=None):
    beta_values = 1.0 / temperatures
    pool = Pool(processes=4)  # Adjust number of processes based on your system
    # Include flag to collect time series only at specified temperatures
    tasks = [(N, beta, steps_eq, steps_mc, Jij, False) for beta in
             beta_values]  # Collect time series for all temperatures for simplicity
    results = pool.map(simulation_task, tasks)
    pool.close()
    pool.join()

    results_dict = {'magnetizations': [], 'susceptibilities': [], 'specific_heats': [], 'energies': [],
                    'time_series': []}
    for mag_mean, energy_mean, susceptibility, specific_heat, time_series in results:
        results_dict['magnetizations'].append(mag_mean)
        results_dict['energies'].append(energy_mean)
        results_dict['susceptibilities'].append(susceptibility)
        results_dict['specific_heats'].append(specific_heat)
        results_dict['time_series'].append(time_series)

    # Find the critical temperature based on the peak of the susceptibility
    susceptibilities = np.array(results_dict['susceptibilities'])
    peaks, _ = find_peaks(susceptibilities)
    Tc = temperatures[peaks[0]] if len(peaks) > 0 else None

    return results_dict, Tc
def plot_to_html_img(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f'<img src="data:image/png;base64,{image_base64}" />'
def generate_html_log(plots, matrices_info, timing_info, Tc, time_series_plots, fc_plots, output_path):
    template_str = '''
    <html>
    <head>
        <title>Ising Model Simulation Log</title>
    </head>
    <body>
        <h1>Ising Model Simulation Log</h1>
        <h2>Estimated Critical Temperature: {{Tc}} K</h2>
        {% for plot in plots %}
            <h2>{{plot.title}}</h2>
            {{plot.img_tag}}
        {% endfor %}
        <h2>Time Series Plots</h2>
        {% for plot in time_series_plots %}
            <h3>{{plot.title}}</h3>
            {{plot.img_tag}}
        {% endfor %}
        <h2>Functional Connectivity Plots</h2>
        {% for plot in fc_plots %}
            <h3>{{plot.title}}</h3>
            {{plot.img_tag}}
        {% endfor %}
        <h2>Matrices Information</h2>
        <ul>
            {% for matrix in matrices_info %}
                <li>{{matrix.name}}: Dimension - {{matrix.dimension}}</li>
            {% endfor %}
        </ul>
        <h2>Timing Information</h2>
        <ul>
            <li>Simulation Time: {{timing_info.simulation}} seconds</li>
            <li>Plotting Time: {{timing_info.plotting}} seconds</li>
        </ul>
    </body>
    </html>
    '''
    template = Template(template_str)
    html_str = template.render(plots=plots, time_series_plots=time_series_plots, fc_plots=fc_plots,
                               matrices_info=matrices_info, timing_info=timing_info, Tc=f"{Tc:.2f}" if Tc else "N/A")
    with open(output_path, "w") as file:  # Adjust file path as needed
        file.write(html_str)
def plot_observables(temperatures, observables, title="Observables vs. Temperature"):
    """
    Plots the physical observables as a function of temperature.

    Parameters:
    - temperatures: list or numpy.ndarray, the range of temperatures.
    - observables: dict, a dictionary containing lists of observables like magnetization,
      energy, susceptibility, and specific heat, keyed by their names.
    - title: str, the overall title of the plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    observable_keys = ['magnetizations', 'energies', 'susceptibilities', 'specific_heats']
    observable_titles = ['Magnetization', 'Energy', 'Susceptibility', 'Specific Heat']

    for ax, key, obs_title in zip(axs, observable_keys, observable_titles):
        ax.plot(temperatures, observables[key], label=obs_title)
        ax.set_xlabel('Temperature')
        ax.set_ylabel(obs_title)
        ax.set_title(obs_title)
        ax.legend()

    plt.tight_layout(pad=3.0)
    plt.suptitle(title)
    plt.subplots_adjust(top=0.9)
    plt.show()
def plot_matrix(spin_matrix, title="Spin Matrix"):
    """
    Plots the state of the spin matrix.

    Parameters:
    - spin_matrix: numpy.ndarray, the spin matrix to be visualized.
    - title: str, the title of the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(spin_matrix, cmap='RdBu')
    fig.colorbar(cax)
    ax.set_title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
# Optimisation
def calculate_correlation(matrix1, matrix2):
    """Calculate Pearson correlation coefficient between two matrices, handling NaNs."""
    # Flatten matrices and remove pairs where either is NaN
    valid_indices = ~(np.isnan(matrix1) | np.isnan(matrix2))
    if np.any(valid_indices):
        return pearsonr(matrix1[valid_indices], matrix2[valid_indices])[0]
    else:
        return np.nan

def calculate_simulated_fc(time_series, fill_value=0):
    time_series = np.asarray(time_series)

    # Define the measure, here Pearson correlation
    connectivity_measure = ConnectivityMeasure(kind='correlation')

    # Compute the matrix, assuming time_series is a 2D numpy array as before
    correlation_matrix = connectivity_measure.fit_transform([time_series])[0]

    # # Visualizing the correlation matrix
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    # plt.title("Functional Connectivity Matrix")
    # plt.show()
    return correlation_matrix
def combined_matrix_distance(A, B, alpha=0.5, beta=0.5, F_max=None):
    """
    Calculates a combined distance metric between two matrices, incorporating
    both the Frobenius norm of their difference and the Pearson correlation
    of their flattened forms using only the elements above the diagonal.

    Parameters:
    - A, B: Input matrices of the same size.
    - alpha, beta: Weights for the Frobenius norm component and the correlation component, respectively.
    - F_max: Optional normalization factor for the Frobenius norm. If None, it's calculated as the maximum possible Frobenius norm given the matrices' range.

    Returns:
    - Combined distance metric as a float.
    """

    # Calculate the Frobenius norm of the difference between matrices
    frobenius_norm = np.linalg.norm(A - B, 'fro')

    # Automatically determine F_max if not provided
    if F_max is None:
        # Assuming the range of matrix elements can be from 0 to 1
        # Adjust this based on your actual data range for more accurate normalization
        max_diff_matrix = np.ones(A.shape)  # Maximum difference matrix
        F_max = np.linalg.norm(max_diff_matrix, 'fro')

    # Normalize Frobenius norm component
    frobenius_component = 1 - frobenius_norm / F_max

    # Get indices for the upper triangle above the diagonal
    triu_indices = np.triu_indices_from(A, k=1)  # k=1 excludes the diagonal

    # Select elements from the flattened matrices using the upper-triangle indices
    A_flat_upper = A[triu_indices]
    B_flat_upper = B[triu_indices]

    # Calculate Pearson correlation coefficient for the upper-triangle elements
    correlation = np.corrcoef(A_flat_upper, B_flat_upper)[0, 1]

    # Combine components into a single metric
    combined_metric = alpha * frobenius_component +(1- beta * correlation)
    # Identify the distance type based on alpha and beta values
    if alpha == 0 and beta == 1:
        distance_type = 'correlation'
    elif alpha == 1 and beta == 0:
        distance_type = 'frobenius_norm'
    else:
        distance_type = 'combined'

    # Return both the combined metric and the distance type
    return combined_metric, distance_type
def remove_diagonal(matrix):
    """Remove the diagonal elements of a matrix by setting them to zero."""
    np.fill_diagonal(matrix, 0)
    return matrix
def discrepancy_function(params, empirical_fc, N, steps_eq, steps_mc, Jij=None, mu=None,global_results=None,global_results_2=None):
    """Calculate discrepancy between empirical and simulated FC."""

    temperature, alpha = params
    if np.isnan(temperature) or np.isnan(alpha):
        # if number is Nan returning a large distance for the loop such that is left out, to prevent Nan falling into a loop
        distance = 1000
        return distance
    beta = 1.0 / temperature
    # Simulate time series with given parameters
    mag_mean, energy_mean, susceptibility, specific_heat,simulated_time_series = simulation_task((N, beta, steps_eq, steps_mc, Jij, mu, alpha, True,global_results_2))
    global_results['temperature'].append(temperature)
    global_results['magnetization'].append(mag_mean)
    global_results['susceptibility'].append(susceptibility)
    global_results['specific_heat'].append(specific_heat)
    global_results['time_series'].append(simulated_time_series)
    global_results['energy'].append(energy_mean)
    # Calculate simulated FC matrix
    simulated_fc = calculate_simulated_fc(simulated_time_series)
    # Calculate distance between empirical and simulated FC (e.g., Frobenius norm of the difference)
    # distance = np.nanmean((empirical_fc - simulated_fc) ** 2)

    # Remove diagonal elements from both empirical and simulated FC matrices
    empirical_fc_no_diag = remove_diagonal(np.copy(empirical_fc))
    simulated_fc_no_diag = remove_diagonal(np.copy(simulated_fc))

    # distance = combined_matrix_distance(empirical_fc, simulated_fc, alpha=0.5, beta=0.5)
    # Calculate distance between the empirical and simulated FC with diagonals removed
    #alpha = 0.5 distance Bias
    #beta = 0.5 coorelation Bias
    # Use the modified combined_matrix_distance to get both distance and type
    distance, distance_type = combined_matrix_distance(empirical_fc_no_diag, simulated_fc_no_diag, alpha=0, beta=1, F_max=None)
    # print("Temperature : ", temperature, " , alpha :", alpha, " , Distance : ", distance)
    # Append data to global_results including the distance type
    global_results['data'].append(
        {'temperature': temperature, 'alpha': alpha, 'distance': distance, 'distance_type': distance_type})
    print(f"Temperature: {temperature}, Alpha: {alpha}, Distance: {distance}, Distance Type: {distance_type}")

    return distance
def load_matrix(filepath, dtype=np.float64):
    extension = filepath.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath, delimiter=',', dtype=dtype)
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)
def extract_rho(path):
    time_series = np.squeeze(load_matrix(path))
    time_series = np.asarray(time_series)
    if time_series.shape[0] < time_series.shape[1]:
        time_series = time_series.transpose()
    # Define the measure, here Pearson correlation
    connectivity_measure = ConnectivityMeasure(kind='correlation')
    # Compute the matrix, assuming time_series is a 2D numpy array as before
    correlation_matrix = connectivity_measure.fit_transform([time_series])[0]

    return correlation_matrix

def optimize_parameters(time_series_path, N, steps_eq, steps_mc, Jij=None, bounds=((0.01, 1.5), (-3, 3)), mu=None,
                        output_folder=None):
    # Extract and save empirical FC
    # Load the empirical time series for comparison
    if time_series_path.endswith('time_series.csv'):
        time_series_emperical = np.loadtxt(time_series_path, delimiter=',')
        np.save(os.path.join(output_folder, 'time_series_emperical.npy'), time_series_emperical)
        # Calculate correlations
        empirical_fc = extract_rho(time_series_path)
    elif time_series_path.endswith('mean_empirical_fc.csv'):
        empirical_fc = np.loadtxt(time_series_path,delimiter=',')
    else:
        raise ValueError("Invalid file format for the empirical time series")

    np.save(os.path.join(output_folder, 'Empirical_fc_matrix_optimized.npy'), empirical_fc)
    # Plot empirical FC matrix
    plt.figure(figsize=(10, 10))
    empirical_fc_no_diag = remove_diagonal(np.copy(empirical_fc))
    sns.heatmap(empirical_fc_no_diag, annot=True, cmap='coolwarm')
    plt.title("Empirical Functional Connectivity Matrix")
    plt.savefig(os.path.join(output_folder, 'Empirical_fc_plot_optimized.png'))
    plt.close()

    # Initialize global results storage
    global_results = {'temperature': [], 'magnetization': [], 'susceptibility': [], 'data': []}


    # Optimize temperature and alpha using dual annealing
    result = dual_annealing(
        discrepancy_function,
        bounds,
        args=(empirical_fc_no_diag, N, steps_eq, steps_mc, Jij, mu, global_results),
        maxiter=500,  # Increase the maximum number of iterations
        maxfun=500,  # Increase the maximum number of function evaluations
        initial_temp=3,  # Adjust the initial temperature if needed
        restart_temp_ratio=2e-3,  # Adjust the restart temperature ratio
        visit=5,  # Increase the visit parameter
        accept=-5.0  # Adjust the acceptance parameter
    )

    # After optimization, convert global_results['data'] to DataFrame and save
    df_results = pd.DataFrame(global_results['data'])
    # Save to CSV
    csv_file_path = os.path.join(output_folder, 'optimization_results.csv')
    df_results.to_csv(csv_file_path, index=False)
    # Save to Pickle
    pickle_file_path = os.path.join(output_folder, 'optimization_results.pkl')
    df_results.to_pickle(pickle_file_path)
    # Calculate Tc from the peak in susceptibility
    generate_and_save_graphs(global_results, output_folder, mu=mu)
    susceptibilities = np.array(global_results['susceptibility'])
    temperatures = np.array(global_results['temperature'])
    min_height = susceptibilities.mean()  # This is arbitrary, adjust based on your data
    prominence = 0.1  # Also arbitrary, adjust based on your data
    peaks, _ = find_peaks(susceptibilities)
    peaks, properties = find_peaks(susceptibilities, height=min_height, prominence=prominence)

    if peaks.size > 0:
        # If multiple peaks are found, choose the one with the highest prominence as the critical point
        # This step assumes the most physically relevant transition point will have the most significant change in susceptibility
        most_prominent_peak_index = peaks[np.argmax(properties["prominences"])]
        Tc_estimated = temperatures[most_prominent_peak_index]
    else:
        Tc_estimated = None


    # Plot observables vs temperature and save to output folder
    plot_observables_vs_temperature(global_results, Tc_estimated, result.x[0], output_folder)
    print("Optimization results:", result.x)
    # Return optimized parameters and estimated Tc
    return result.x
# def calculate_matrix_correlations(empirical_fc, simulated_fc, jij):
#     """
#     Calculates and returns the correlations between empirical FC and simulated FC,
#     empirical FC and Jij, and simulated FC and Jij.
#
#     Parameters:
#     - empirical_fc: numpy array representing the empirical functional connectivity matrix.
#     - simulated_fc: numpy array representing the simulated functional connectivity matrix.
#     - jij: numpy array representing the Jij structural connectivity matrix.
#
#     Returns:
#     - A dictionary containing the correlation coefficients between:
#       empirical and simulated FC, empirical FC and Jij, and simulated FC and Jij.
#     """
#
#     # Flatten the matrices to simplify correlation calculations
#     empirical_fc_flat = empirical_fc.flatten()
#     simulated_fc_flat = simulated_fc.flatten()
#     jij_flat = jij.flatten()
#
#     # Calculate the correlations
#     corr_empirical_simulated = np.corrcoef(empirical_fc_flat, simulated_fc_flat)[0, 1]
#     corr_empirical_jij = np.corrcoef(empirical_fc_flat, jij_flat)[0, 1]
#     corr_simulated_jij = np.corrcoef(simulated_fc_flat, jij_flat)[0, 1]
#
#     # Return the results in a dictionary
#     correlations = {
#         'empirical_simulated': corr_empirical_simulated,
#         'empirical_jij': corr_empirical_jij,
#         'simulated_jij': corr_simulated_jij
#     }
#
#     return correlations
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

def calculate_matrix_correlations_and_norms_upper(empirical_fc, simulated_fc, jij):
    """
    Calculates and returns the correlations and Frobenius norms between the upper triangular parts
    (excluding the diagonal) of empirical FC and simulated FC, empirical FC and Jij,
    and simulated FC and Jij.

    Parameters:
    - empirical_fc: numpy array representing the empirical functional connectivity matrix.
    - simulated_fc: numpy array representing the simulated functional connectivity matrix.
    - jij: numpy array representing the Jij structural connectivity matrix.

    Returns:
    - A dictionary containing the correlation coefficients and Frobenius norms between:
      the upper triangular parts (excluding the diagonal) of empirical and simulated FC,
      empirical FC and Jij, and simulated FC and Jij.
    """

    # Indices of the upper triangle, excluding the diagonal
    upper_tri_indices = np.triu_indices_from(empirical_fc, k=1)

    # Select upper triangular parts, excluding the diagonal
    empirical_fc_upper = empirical_fc[upper_tri_indices]
    simulated_fc_upper = simulated_fc[upper_tri_indices]
    jij_upper = jij[upper_tri_indices]

    # Calculate the correlations
    corr_empirical_simulated = np.corrcoef(empirical_fc_upper, simulated_fc_upper)[0, 1]
    corr_empirical_jij = np.corrcoef(empirical_fc_upper, jij_upper)[0, 1]
    corr_simulated_jij = np.corrcoef(simulated_fc_upper, jij_upper)[0, 1]

    # Calculate Frobenius norms for the differences in upper triangular parts
    # First, create matrices of differences for the upper triangles
    diff_empirical_simulated_upper = np.zeros_like(empirical_fc)
    diff_empirical_jij_upper = np.zeros_like(empirical_fc)
    diff_simulated_jij_upper = np.zeros_like(empirical_fc)

    # Populate the upper triangles of the difference matrices
    diff_empirical_simulated_upper[upper_tri_indices] = empirical_fc_upper - simulated_fc_upper
    diff_empirical_jij_upper[upper_tri_indices] = empirical_fc_upper - jij_upper
    diff_simulated_jij_upper[upper_tri_indices] = simulated_fc_upper - jij_upper

    # Calculate Frobenius norms for these differences
    fro_norm_empirical_simulated = np.linalg.norm(diff_empirical_simulated_upper, 'fro')
    fro_norm_empirical_jij = np.linalg.norm(diff_empirical_jij_upper, 'fro')
    fro_norm_simulated_jij = np.linalg.norm(diff_simulated_jij_upper, 'fro')

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
def remove_diagonal(matrix):
    """Remove the diagonal elements of a matrix by setting them to zero."""
    np.fill_diagonal(matrix, 0)
    return matrix
def optimize_and_simulate(time_series_path, N, steps_eq, steps_mc, output_folder, Jij=None,mu=None):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: Optimization
    optimized_params = optimize_parameters(time_series_path, N, steps_eq, steps_mc, Jij, bounds=((0.0001, 1.5), (-3, 3)),mu=mu,output_folder=output_folder)
    temp_optimized, alpha_optimized = optimized_params

    # Convert optimized temperature to beta
    beta_optimized = 1.0 / temp_optimized

    # Step 2: Simulation with Optimized Parameters
    simulation_start_time = time.time()
    _, _, _, _, time_series_optimized = simulation_task(
        (N, beta_optimized, steps_eq, steps_mc, Jij, None, alpha_optimized, True))
    simulation_end_time = time.time()

    # Load the empirical time series for comparison
    if time_series_path.endswith('time_series.csv'):
        time_series_emperical = np.loadtxt(time_series_path, delimiter=',')
        np.save(os.path.join(output_folder, 'time_series_emperical.npy'), time_series_emperical)
        # Calculate correlations
        empirical_fc = extract_rho(time_series_path)
        empirical_fc_no_diag = remove_diagonal(np.copy(empirical_fc))
    elif time_series_path.endswith('mean_empirical_fc.csv'):
        empirical_fc = np.loadtxt(time_series_path,delimiter=',')
        empirical_fc_no_diag = remove_diagonal(np.copy(empirical_fc))
    else:
        raise ValueError("Invalid file format for the empirical time series")

    time_series = np.asarray(time_series_optimized)
    np.save(os.path.join(output_folder, 'Simulated_time_series_optimized.npy'), time_series)

    # Step 3: Calculate Functional Connectivity
    fc_matrix_optimized = calculate_simulated_fc(time_series_optimized)

    simulated_fc_no_diag = remove_diagonal(np.copy(fc_matrix_optimized))



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
    sns.heatmap(simulated_fc_no_diag, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Simulated Functional Connectivity Matrix")
    plt.savefig(os.path.join(output_folder, 'Simulated_fc_plot_optimized.png'))

    if Jij is not None:
        Jij= normalize_matrix(Jij)
        plt.close()
        plt.figure(figsize=(10, 10))
        sns.heatmap(Jij, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Jij- Structural Connectivity")
        plt.savefig(os.path.join(output_folder, 'Jij.png'))


        # Calculate correlations and Frobenius norms
        results = calculate_matrix_correlations_and_norms_upper(empirical_fc_no_diag, simulated_fc_no_diag, Jij)

        # Print the correlation results and Frobenius norms with evaluations
        print_results_with_evaluation(results)
        print(" ")

    # Generate and Save HTML Log
    timing_info = {
        'simulation': simulation_end_time - simulation_start_time,
        'plotting': time.time() - simulation_end_time
    }
    log_path = os.path.join(output_folder, 'simulation_log.html')
    generate_html_log([], [{'name': 'FC Matrix Optimized', 'dimension': fc_matrix_optimized.shape}], timing_info,
                      temp_optimized, [], [], output_path=log_path)


    return optimized_params, results,os.path.join(output_folder, 'fc_matrix_optimized.npy'), os.path.join(output_folder,
                                                                                                  'time_series_optimized.npy'), log_path


############################# Other Functions #############################
import os
import logging
import pandas as pd
from scipy.stats import pearsonr, f_oneway
from concurrent.futures import ProcessPoolExecutor # For parallel processing
import concurrent.futures as cf

def write_values_to_file(output_folder, description_array, value_array):
    """
    Writes descriptions and their corresponding values to a text file.

    Parameters:
    - output_folder: str, the path to the folder where the output file will be saved.
    - description_array: list of str, the descriptions to be written to the file.
    - value_array: list, the values corresponding to each description.

    The function saves a file named 'output.txt' in the specified output folder,
    writing each description and its corresponding value in the format 'description: value'.
    """

    # Ensure the output folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the output file path
    file_path = os.path.join(output_folder, 'output.txt')

    # Open the file for writing
    with open(file_path, 'w') as file:
        # Iterate over both arrays simultaneously
        for description, value in zip(description_array, value_array):
            # Write the description and value to the file
            file.write(f'{description}: {value}\n')

    print(f'Values have been written to {file_path}')
def perform_statistical_analysis(df):
    # Check for the existence of required columns before proceeding
    required_columns = ['empirical_simulated', 'empirical_jij', 'simulated_jij']
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        logging.warning(f"Missing columns in DataFrame: {missing_columns}. Analysis may be limited.")
    else:
        # Proceed with analysis if all columns are present
        correlations = df[required_columns]
        print("Correlations among metrics:")
        print(correlations.corr())

    # Example of handling missing data gracefully
    if 'subject_id' in df.columns and 'empirical_simulated' in df.columns:
        subjects = df['subject_id'].unique()
        if len(subjects) > 1:
            anova_data = [df[df['subject_id'] == subject]['empirical_simulated'].values for subject in subjects]
            anova_result = f_oneway(*anova_data)
            print(f"ANOVA result for differences between subjects in empirical_simulated: F={anova_result.statistic}, p={anova_result.pvalue}")
        else:
            logging.warning("Insufficient subjects for ANOVA.")
    else:
        logging.warning("Missing 'subject_id' or 'empirical_simulated' column for ANOVA analysis.")
def save_to_csv(df, output_path):
    # Save the DataFrame to a CSV file for further analysis or record-keeping
    df.to_csv(output_path, index=False)
def create_individual_output_folders(base_output_folder, subject_id, parcellation):
    """
    Creates individualized output folders for a given subject and parcellation combination
    within a specified base output folder.

    Parameters:
    - base_output_folder: The root directory where output folders will be created.
    - subject_id: The identifier for the subject.
    - parcellation: The name of the parcellation scheme being used.

    The function constructs a path using these parameters and ensures the directory exists.
    """
    # Construct the path for the new output folder
    output_path = os.path.join(base_output_folder, subject_id, parcellation)

    # Create the directory if it does not already exist
    os.makedirs(output_path, exist_ok=True)

    # Optionally, return the path of the created folder for further use
    return output_path

def read_and_process_files(subject_path, parcellation, base_output_folder):
    """
    Function to read files for a given subject and parcellation, perform calculations, and tabulate results.
    """
    results = []  # List to store results for tabulation
    parcellation_path = os.path.join(subject_path, parcellation)
    fmri_path = pet_path = jij_path = None
    try:
        for file_name in os.listdir(parcellation_path):
            file_path = os.path.join(parcellation_path, file_name)
            if file_name.endswith('time_series.csv'):
                fmri_path = file_path
            elif file_name.endswith('mean_empirical_fc.csv'):
                fmri_path = file_path
            elif file_name.endswith('features.txt'):
                pet_path = file_path
            elif file_name.endswith('mean_pet.csv'):
                pet_path = file_path
            elif file_name.endswith('features.csv'):
                pet_path = file_path
            elif file_name.endswith('Jij.csv'):
                jij_path = file_path
            elif file_name.endswith('mean_jij.csv'):
                jij_path = file_path

        if fmri_path and pet_path and jij_path:
            # Process the files for this specific subject and parcellation
            subject_id = os.path.basename(subject_path)
            optimized_values = process_files(fmri_path, pet_path, jij_path, subject_id, parcellation, base_output_folder)
            # Record optimized values with subject and parcellation identifiers
            results.append((subject_id, parcellation, optimized_values))
    except Exception as e:
        logging.error(f"Error accessing files in {parcellation_path}: {e}")
        return None

    # Create a DataFrame from the results, if any
    if results:
        df = pd.DataFrame(results, columns=['SubjectID', 'Parcellation', 'OptimizedValues'])
        return df
    else:
        return None
def record_and_analyze(base_folder, base_output_folder):
    all_data = []  # Container for all subjects' optimized values and metrics

    try:
        subjects = os.listdir(base_folder)
    except FileNotFoundError:
        logging.error(f"The directory {base_folder} was not found.")
        return

    for subject_id in subjects:
        subject_path = os.path.join(base_folder, subject_id)
        if os.path.isdir(subject_path):
            for parcellation in os.listdir(subject_path):
                # Process each subject and parcellation combination
                df = read_and_process_files(subject_path, parcellation, base_output_folder)
                if df is not None:
                    all_data.extend(df.to_dict('records'))

    # Convert to DataFrame for easier manipulation
    if all_data:
        df = pd.DataFrame(all_data)
        # Perform statistical analysis
        perform_statistical_analysis(df)  # Placeholder for actual analysis function
        return df
    else:
        return None
def record_and_analyze_parallel(base_folder, base_output_folder, max_workers=4):
    all_data = []  # Container for all subjects' optimized values and metrics

    try:
        subjects = os.listdir(base_folder)
    except FileNotFoundError:
        logging.error(f"The directory {base_folder} was not found.")
        return

    # Create a process pool executor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for subject_id in subjects:
            subject_path = os.path.join(base_folder, subject_id)
            if os.path.isdir(subject_path):
                for parcellation in os.listdir(subject_path):
                    # Submit each subject and parcellation combination as a separate task
                    futures.append(
                        executor.submit(read_and_process_files, subject_path, parcellation, base_output_folder))

        # Wait for all submitted tasks to complete and collect their results
        for future in cf.as_completed(futures):
            df = future.result()
            if df is not None:
                all_data.extend(df.to_dict('records'))

    # Convert to DataFrame for easier manipulation
    if all_data:
        df = pd.DataFrame(all_data)
        # Perform statistical analysis
        perform_statistical_analysis(df)  # Placeholder for actual analysis function
        return df
    else:
        return None

def plot_observables_vs_temperature(global_results, Tc, optimal_temperature, output_folder):
    # Ensure the directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Sort the arrays
    sorted_indices = np.argsort(global_results['temperature'])
    sorted_temperature = np.array(global_results['temperature'])[sorted_indices]
    sorted_magnetization = np.array(global_results['magnetization'])[sorted_indices]
    sorted_susceptibility = np.array(global_results['susceptibility'])[sorted_indices]

    # Start plotting
    plt.figure(figsize=(12, 6))

    # Plot Magnetization
    plt.subplot(1, 2, 1)
    plt.plot(sorted_temperature, sorted_magnetization, marker='o', linestyle='', label='Magnetization')
    plt.axvline(x=Tc, color='r', linestyle='--', label=f'Estimated Tc={Tc}')
    plt.axvline(x=optimal_temperature, color='g', linestyle='--', label=f'Optimal Temperature={optimal_temperature}')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    plt.legend()

    # Plot Susceptibility
    plt.subplot(1, 2, 2)
    plt.plot(sorted_temperature, sorted_susceptibility, marker='o', linestyle='', label='Susceptibility')
    plt.axvline(x=Tc, color='r', linestyle='--', label=f'Estimated Tc={Tc}')
    plt.axvline(x=optimal_temperature, color='g', linestyle='--', label=f'Optimal Temperature={optimal_temperature}')
    plt.xlabel('Temperature')
    plt.ylabel('Susceptibility')
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(output_folder, 'observables_vs_temperature.png')
    plt.savefig(output_path)
    plt.close()  # Close figure after saving to free up memory

    print(f"Observables vs. Temperature plot saved to {output_path}")

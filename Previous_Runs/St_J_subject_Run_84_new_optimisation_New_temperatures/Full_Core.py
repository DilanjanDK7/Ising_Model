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
import os


# # Ensure reproducibility
# np.random.seed(42)


# def initialize_spin_matrix(N):
#     return np.random.choice([-1, 1], size=(N, N))
def initialize_spin_matrix(N):
    return np.random.choice([-1, 1], size=(N))


# Modified metropolis_step to include mu and alpha
def metropolis_step(spin_array, beta, Jij=None, mu=None, alpha=1.0, J_default=1.0):
    N = spin_array.shape[0]  # Total number of spins

    # Function to calculate delta_E for nearest neighbors if Jij is not provided
    def delta_E_nearest_neighbors(i):
        return 2 * J_default * spin_array[i] * (
                spin_array[(i - 1) % N] + spin_array[(i + 1) % N])

    for _ in range(N):
        i = np.random.randint(0, N)  # Randomly select a spin
        local_T = 1 / beta * (mu[i] ** alpha) if mu is not None else 1 / beta
        local_beta = 1 / local_T

        if Jij is not None:
            # Calculate the change in energy if spin i were flipped, using Jij for interactions
            delta_E = 0
            for j in range(N):  # Loop over all spins to account for interactions
                if Jij[i, j] != 0:  # Only add to delta_E if there's an interaction
                    delta_E += 2 * Jij[i, j] * spin_array[i] * spin_array[j]
        else:
            # Calculate the change in energy for nearest neighbors if Jij is not provided
            delta_E = delta_E_nearest_neighbors(i)

        # Decide whether to flip the spin
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E * local_beta):
            spin_array[i] *= -1

    return spin_array


# def normalize_matrix(X):
#     """
#     Normalize each column of the matrix X to have values between 0 and 1,
#     preserving zero values.
#
#     Parameters:
#     - X: A numpy array of shape (n, n).
#
#     Returns:
#     - X_norm: The normalized numpy array of shape (n, n).
#     """
#     # Copy X to preserve original data and avoid in-place modifications
#     X_norm = X.copy().astype(float)  # Ensure the type is float for division
#
#     # Calculate the min and max of each column
#     mins = np.min(X, axis=0)
#     maxs = np.max(X, axis=0)
#
#     # Calculate the range (max-min) of each column, avoiding division by zero
#     ranges = maxs - mins
#     ranges[ranges == 0] = 1  # Avoid division by zero
#
#     # Normalize each column to 0-1 range, preserving zeros
#     for i in range(X.shape[1]):  # Iterate through each column
#         # Avoid changing zero values - mask non-zero (X > 0) values only
#         mask = X[:, i] > 0
#         X_norm[mask, i] = (X[mask, i] - mins[i]) / ranges[i]
#
#     return X_norm
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
        return 2 * J_default * spin_array[i] * (
                spin_array[(i - 1) % N] + spin_array[(i + 1) % N])

    for i in range(N):  # Iterate over each spin in sequence
        # local_T = 1 / beta * (mu[i] ** alpha) if mu is not None else 1 / beta
        local_T = (mu[i] ** alpha) / beta if mu is not None else 1 / beta
        local_beta = 1 / local_T

        if Jij is not None:
            # Normalize the matrix jij
            # Jij_normalised = normalize_matrix(Jij)
            Jij_normalised = Jij
            # Calculate the change in energy if spin i were flipped, using Jij for interactions
            delta_E = 0
            for j in range(N):  # Loop over all spins to account for interactions
                delta_E += 2 * Jij_normalised[i, j] * spin_array[i] * spin_array[j]
        else:
            # Calculate the change in energy for nearest neighbors if Jij is not provided
            delta_E = delta_E_nearest_neighbors(i)
        # Decide whether to flip the spin
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E * local_beta):
            # print("<",np.exp(-delta_E * local_beta))
            # print(np.random.rand() < np.exp(-delta_E * local_beta))
            spin_array[i] *= -1

    return spin_array


'''
def metropolis_step(spin_matrix, beta, Jij=None, mu=None, alpha=1.0):
    N = spin_matrix.shape[0]
    for _ in range(N ** 2):
        x, y = np.random.randint(0, N, size=2)
        local_T = 1 / beta * (mu[x] ** alpha) if mu is not None else 1 / beta
        local_beta = 1 / local_T
        delta_E = 2 * spin_matrix[x, y] * (
            np.sum(Jij[x, y] * spin_matrix) if Jij is not None else
            (spin_matrix[(x + 1) % N, y] + spin_matrix[x, (y + 1) % N] +
             spin_matrix[(x - 1) % N, y] + spin_matrix[x, (y - 1) % N]))
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E * local_beta):
            spin_matrix[x, y] *= -1
    return spin_matrix
'''
'''
def calculate_observables(spin_matrix, beta, Jij=None):
    N = spin_matrix.shape[0]
    mag = np.abs(np.sum(spin_matrix)) / N ** 2
    energy = -np.sum(spin_matrix * (
            np.roll(spin_matrix, 1, axis=0) +
            np.roll(spin_matrix, -1, axis=0) +
            np.roll(spin_matrix, 1, axis=1) +
            np.roll(spin_matrix, -1, axis=1)) / 4.0) if Jij is None else \
        -np.sum(spin_matrix.flatten() * np.dot(Jij.reshape(N , N * N), spin_matrix.flatten())) / 2.0
    return mag, energ
'''

'''
def calculate_observables(spin_matrix, beta, Jij=None):
    N = spin_matrix.shape[0]
    # Calculate magnetization
    mag = np.abs(np.sum(spin_matrix)) / (N ** 2)

    if Jij is None:
        # Calculate energy assuming nearest-neighbor interactions
        energy = -np.sum(spin_matrix * (
                np.roll(spin_matrix, 1, axis=0) +
                np.roll(spin_matrix, -1, axis=0) +
                np.roll(spin_matrix, 1, axis=1) +
                np.roll(spin_matrix, -1, axis=1)) / 4.0)
    else:
        # Direct calculation of energy using Jij for given interactions
        # Initialize energy
        energy = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        # Calculate interaction energy contribution
                        energy += -Jij[i, j] * spin_matrix[i, j] * spin_matrix[k, l]

        # Adjust for double-counting
        energy = energy / 2.0

    return mag, energy
'''

import numpy as np


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
    N, beta, steps_eq, steps_mc, Jij, mu, alpha, collect_time_series = params
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


"""
def calculate_empirical_fc(time_series):
    # Calculate empirical functional connectivity, handling NaNs.
    num_sites = time_series[0].shape[0]
    fc_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(i, num_sites):
            series_i = np.array([ts.flatten()[i] for ts in time_series])
            series_j = np.array([ts.flatten()[j] for ts in time_series])
            correlation = calculate_correlation(series_i, series_j)
            fc_matrix[i, j] = fc_matrix[j, i] = correlation
    return fc_matrix
"""

'''
def calculate_empirical_fc(time_series):
    """
    Calculate empirical functional connectivity, handling NaNs, for a time series array of shape N x timepoints.

    Parameters:
    - time_series (numpy.ndarray): A 2D array with shape N x timepoints, where N is the number of sites and
      timepoints is the number of observations over time.

    Returns:
    - fc_matrix (numpy.ndarray): A 2D array representing the functional connectivity matrix, with shape N x N.
    """
    time_series = np.asarray(time_series)
    if time_series.shape[0]>time_series.shape[1]:
        #Assuming the time points are larger than the no of parcellations
        print("Transposing the time series")
        time_series=time_series.T
    else:
        print("The format is parcellations vs time points")
        time_series = time_series

    num_sites = time_series.shape[0]  # Number of sites or regions
    fc_matrix = np.zeros((num_sites, num_sites))

    for i in range(num_sites):
        for j in range(i, num_sites):
            series_i = time_series[i, :]
            series_j = time_series[j, :]

            # Calculate correlation, handling NaN values appropriately
            valid_indices = ~np.isnan(series_i) & ~np.isnan(
                series_j)  # Indices where neither series_i nor series_j is NaN
            if np.any(valid_indices):  # Ensure there's at least some data to correlate
                correlation = np.corrcoef(series_i[valid_indices], series_j[valid_indices])[0, 1]
            else:
                correlation = np.nan  # Set correlation to NaN if no valid data points are found

            fc_matrix[i, j] = fc_matrix[j, i] = correlation

    return fc_matrix
'''

'''
def calculate_empirical_fc(time_series, fill_value=0):
    """
    Calculate empirical functional connectivity, replacing NaNs with a specified fill value for a time series array of shape N x timepoints.

    Parameters:
    - time_series (numpy.ndarray): A 2D array with shape N x timepoints, where N is the number of sites and
      timepoints is the number of observations over time.
    - fill_value (float): The value to use in place of NaNs in the FC matrix. Default is 0.

    Returns:
    - fc_matrix (numpy.ndarray): A 2D array representing the functional connectivity matrix, with shape N x N, with NaNs replaced by fill_value.
    """
    time_series = np.asarray(time_series)
    if time_series.shape[0] > time_series.shape[1]:
        print("Transposing the time series for correct orientation (timepoints vs parcellations).")
        time_series = time_series.T

    num_sites = time_series.shape[0]
    fc_matrix = np.zeros((num_sites, num_sites))

    for i in range(num_sites):
        for j in range(i, num_sites):
            series_i = time_series[i, :]
            series_j = time_series[j, :]

            valid_indices = ~np.isnan(series_i) & ~np.isnan(series_j)
            if np.any(valid_indices):
                correlation = np.corrcoef(series_i[valid_indices], series_j[valid_indices])[0, 1]
            else:
                correlation = fill_value  # Use fill_value instead of np.nan

            fc_matrix[i, j] = fc_matrix[j, i] = correlation

    return fc_matrix
"""def calculate_empirical_fc(time_series):
    num_sites = time_series[0].shape[0] * time_series[0].shape[1]
    fc_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(i, num_sites):
            series_i = np.array([ts.flatten()[i] for ts in time_series])
            series_j = np.array([ts.flatten()[j] for ts in time_series])
            correlation = calculate_correlation(series_i, series_j)
            fc_matrix[i, j] = fc_matrix[j, i] = correlation
    return fc_matrix"""

'''


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


'''
def calculate_empherical_fc(time_series,fill_value=0):
    time_series = np.asarray(time_series)

    # Define the measure, here Pearson correlation
    connectivity_measure = ConnectivityMeasure(kind='correlation')

    # Compute the matrix, assuming time_series is a 2D numpy array as before
    correlation_matrix = connectivity_measure.fit_transform([time_series])[0]

    # Visualizing the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Functional Connectivity Matrix")
    plt.show()
    return correlation_matrix
'''

'''
def combined_matrix_distance(A, B, alpha=0.5, beta=0.5, F_max=None):
    """
    Calculates a combined distance metric between two matrices, incorporating
    both the Frobenius norm of their difference and the Pearson correlation
    of their flattened forms.

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

    # Flatten matrices and calculate Pearson correlation coefficient
    A_flat = A.flatten()
    B_flat = B.flatten()
    correlation = np.corrcoef(A_flat, B_flat)[0, 1]

    # Combine components into a single metric
    combined_metric = alpha * frobenius_component - beta * abs(correlation)

    return combined_metric
'''

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
    combined_metric = alpha * frobenius_component +(1- beta * abs(correlation))

    return combined_metric

def remove_diagonal(matrix):
    """Remove the diagonal elements of a matrix by setting them to zero."""
    np.fill_diagonal(matrix, 0)
    return matrix
def discrepancy_function(params, empirical_fc, N, steps_eq, steps_mc, Jij=None, mu=None):
    """Calculate discrepancy between empirical and simulated FC."""

    temperature, alpha = params
    if np.isnan(temperature) or np.isnan(alpha):
        # if number is Nan returning a large distance for the loop such that is left out, to prevent Nan falling into a loop
        distance = 1000
        return distance
    beta = 1.0 / temperature
    # Simulate time series with given parameters
    print("Temperature : ", temperature, " , alpha :", alpha)
    _, _, _, _, simulated_time_series = simulation_task((N, beta, steps_eq, steps_mc, Jij, mu, alpha, True))
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
    distance = combined_matrix_distance(empirical_fc_no_diag, simulated_fc_no_diag, alpha=0, beta=1)

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


'''
def extract_rho(path):
        time_series = np.squeeze(load_matrix(path))

        shape_ts = time_series.shape

        assert len(shape_ts) == 2
        assert shape_ts[0] != shape_ts[1]

        if shape_ts[0] > shape_ts[1]:
            time_series = time_series.T

        empirical_fc = np.corrcoef(time_series)

        return empirical_fc

'''


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


def optimize_parameters(time_series_path, N, steps_eq, steps_mc, Jij=None, bounds=((0.01, 10), (-3, 3)), mu=None,
                        output_folder=None):
    empirical_fc = extract_rho(time_series_path)
    np.save(os.path.join(output_folder, 'Empherical_fc_matrix_optimized.npy'), empirical_fc)
    # plot_fc = plot_functional_connectivity(empirical_fc)
    # plot_fc.savefig(os.path.join(output_folder, 'Empherical_fc_plot_optimized.png'))

    empirical_fc_no_diag = remove_diagonal(np.copy(empirical_fc))

    # Visualizing the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(empirical_fc_no_diag, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title("Functional Connectivity Matrix")
    plt.savefig(os.path.join(output_folder, 'Empherical_fc_plot_optimized.png'))

    """Find optimal temperature and alpha using dual annealing."""
    result = dual_annealing(discrepancy_function, bounds, args=(empirical_fc_no_diag, N, steps_eq, steps_mc, Jij, mu),
                            maxfun=1e2)
    return result.x  # Returns the optimal temperature and alpha


'''
def main():
    N = 20  # System size
    temperatures = np.linspace(2.0, 3.0, 50)  # Temperature range
    steps_eq = 1000  # Equilibration steps
    steps_mc = 1000  # MC steps for measurements

    start_time = time.time()
    results, Tc = run_simulation_parallel(N, temperatures, steps_eq, steps_mc, Jij=None, Tc=None, specific_temp=None)
    simulation_time = time.time() - start_time

    # Generate plots for observables
    fig_magnetization = plot_observables(temperatures, results['magnetizations'], 'Temperature', 'Magnetization',
                                         'Magnetization vs Temperature')
    fig_susceptibility = plot_observables(temperatures, results['susceptibilities'], 'Temperature', 'Susceptibility',
                                          'Susceptibility vs Temperature')
    fig_specific_heat = plot_observables(temperatures, results['specific_heats'], 'Temperature', 'Specific Heat',
                                         'Specific Heat vs Temperature')
    fig_energy = plot_observables(temperatures, results['energies'], 'Temperature', 'Energy', 'Energy vs Temperature')

    # Generate time series and functional connectivity plots for selected temperatures
    time_series_plots = []
    fc_plots = []
    for i, T in enumerate(temperatures):
        fig_ts = plot_time_series(results['time_series'][i])
        time_series_plots.append({'title': f'Time Series at T={T:.2f}', 'img_tag': plot_to_html_img(fig_ts)})

        fc_matrix = calculate_functional_connectivity(results['time_series'][i])
        fig_fc = plot_matrix(fc_matrix)
        fc_plots.append({'title': f'Functional Connectivity at T={T:.2f}', 'img_tag': plot_to_html_img(fig_fc)})

    plotting_time = time.time() - start_time - simulation_time

    # Compile information for HTML log
    plots = [
        {'title': 'Magnetization vs Temperature', 'img_tag': plot_to_html_img(fig_magnetization)},
        {'title': 'Susceptibility vs Temperature', 'img_tag': plot_to_html_img(fig_susceptibility)},
        {'title': 'Specific Heat vs Temperature', 'img_tag': plot_to_html_img(fig_specific_heat)},
        {'title': 'Energy vs Temperature', 'img_tag': plot_to_html_img(fig_energy)}
    ]
    matrices_info = [{'name': 'Functional Connectivity Matrix', 'dimension': f'{N} x {N}'}]
    timing_info = {'simulation': f"{simulation_time:.2f}", 'plotting': f"{plotting_time:.2f}"}

    generate_html_log(plots, matrices_info, timing_info, Tc, time_series_plots, fc_plots)


if __name__ == "__main__":
    main()

'''

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



def remove_diagonal(matrix):
    """Remove the diagonal elements of a matrix by setting them to zero."""
    np.fill_diagonal(matrix, 0)
    return matrix
def optimize_and_simulate(time_series_path, N, steps_eq, steps_mc, output_folder, Jij=None,mu=None):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: Optimization
    optimized_params = optimize_parameters(time_series_path, N, steps_eq, steps_mc, Jij, bounds=((0.001, 3), (-3, 3)),mu=mu,output_folder=output_folder)
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
    sns.heatmap(simulated_fc_no_diag, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title("Simulated Functional Connectivity Matrix")
    plt.savefig(os.path.join(output_folder, 'Simulated_fc_plot_optimized.png'))

    if Jij is not None:
        Jij= normalize_matrix(Jij)
        plt.close()
        plt.figure(figsize=(10, 10))
        sns.heatmap(Jij, annot=False, fmt=".2f", cmap='coolwarm')
        plt.title("Jij- Structural Connectivity")
        plt.savefig(os.path.join(output_folder, 'Jij.png'))
        # Calculate correlations
        empirical_fc = extract_rho(time_series_path)
        empirical_fc_no_diag = remove_diagonal(np.copy(empirical_fc))

        # Calculate correlations and Frobenius norms
        results = calculate_matrix_correlations_and_norms(empirical_fc_no_diag, simulated_fc_no_diag, Jij)

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


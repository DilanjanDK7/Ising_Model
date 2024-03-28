import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import time
from jinja2 import Template
from scipy.signal import find_peaks
from multiprocessing import Pool
from scipy.optimize import dual_annealing
import numpy as np
from scipy.stats import pearsonr
import scipy
import os

# # Ensure reproducibility
# np.random.seed(42)


# def initialize_spin_matrix(N):
#     return np.random.choice([-1, 1], size=(N, N))
def initialize_spin_matrix(N):
    return np.random.choice([-1, 1], size=(N))


# Modified metropolis_step to include mu and alpha
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


def simulation_task(params):
    N, beta, steps_eq, steps_mc, Jij, mu, alpha, collect_time_series = params
    spin_matrix = initialize_spin_matrix(N)
    for _ in range(steps_eq):
        metropolis_step(spin_matrix, beta, Jij, mu, alpha)

    mags, energies, time_series = [], [], []
    for _ in range(steps_mc):
        metropolis_step(spin_matrix, beta, Jij)
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


def calculate_empirical_fc(time_series):
    """Calculate empirical functional connectivity, handling NaNs."""
    num_sites = time_series[0].shape[0]
    fc_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(i, num_sites):
            series_i = np.array([ts.flatten()[i] for ts in time_series])
            series_j = np.array([ts.flatten()[j] for ts in time_series])
            correlation = calculate_correlation(series_i, series_j)
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

def discrepancy_function(params, empirical_fc, N, steps_eq, steps_mc, Jij=None,mu=None):
    """Calculate discrepancy between empirical and simulated FC."""

    temperature, alpha = params
    if np.isnan(temperature) or np.isnan(alpha):
        #if number is Nan returning a large distance for the loop such that is left out, to prevent Nan falling into a loop
        distance = 1000
        return distance
    beta = 1.0 / temperature
    # Simulate time series with given parameters
    print("Temperature : ",temperature," , alpha :", alpha)
    _, _, _, _, simulated_time_series = simulation_task((N, beta, steps_eq, steps_mc, Jij, mu, alpha, True))
    simulated_fc = calculate_empirical_fc(simulated_time_series)
    # Calculate distance between empirical and simulated FC (e.g., Frobenius norm of the difference)
    distance = np.nanmean((empirical_fc - simulated_fc) ** 2)
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

        shape_ts = time_series.shape

        assert len(shape_ts) == 2
        assert shape_ts[0] != shape_ts[1]

        if shape_ts[0] > shape_ts[1]:
            time_series = time_series.T

        empirical_fc = np.corrcoef(time_series)

        return empirical_fc
def optimize_parameters(time_series_path, N, steps_eq, steps_mc, Jij=None, bounds=((0.01, 10), (-3, 3)),mu=None,output_folder=None):

    # if output_folder=None
    #
    # else:
    #     print(out)



    empirical_fc = extract_rho(time_series_path)
    np.save(os.path.join(output_folder, 'Empherical_fc_matrix_optimized.npy'), empirical_fc)
    plot_fc = plot_functional_connectivity(empirical_fc)
    plot_fc.savefig(os.path.join(output_folder, 'Empherical_fc_plot_optimized.png'))

    """Find optimal temperature and alpha using dual annealing."""
    result = dual_annealing(discrepancy_function, bounds, args=(empirical_fc, N, steps_eq, steps_mc, Jij,mu),maxfun=1e2)
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
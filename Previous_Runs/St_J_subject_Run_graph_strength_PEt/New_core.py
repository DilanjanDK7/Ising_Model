from Full_Core import *
import numpy as np
import os
import time
from scipy.optimize import dual_annealing
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from multiprocessing import Pool


def normalize_matrix(matrix):
    return matrix / np.max(np.abs(matrix))


def metropolis_step_all_spins(spin_array, beta, Jij=None, mu=None, alpha=1.0, gamma=1.0):
    N = spin_array.shape[0]
    if Jij is not None:
        Jij = normalize_matrix(Jij)
        graph_strength = np.sum(Jij, axis=1)

    for i in range(N):
        local_T = 1 / beta
        if mu is not None and Jij is not None:
            local_T = ((mu[i] ** alpha) + (graph_strength[i] ** gamma)) / (2 * beta)
        elif mu is not None:
            local_T = (mu[i] ** alpha) / beta
        elif Jij is not None:
            local_T = (graph_strength[i] ** gamma) / beta
        local_beta = 1 / local_T

        delta_E = 0
        if Jij is not None:
            for j in range(N):
                if i != j:
                    delta_E += 2 * Jij[i, j] * spin_array[i] * spin_array[j]
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E * local_beta):
            spin_array[i] *= -1
    return spin_array





def optimize_parameters(empirical_fc, N, steps_eq, steps_mc, Jij=None, mu=None,
                        bounds=((0.01, 10), (-3, 3), (0.5, 2.0))):
    def objective_function(params):
        temperature, alpha, gamma = params
        beta = 1.0 / temperature
        simulated_ts = simulation_task((N, beta, steps_eq, steps_mc, Jij, mu, alpha, gamma, True))
        connectivity_measure = ConnectivityMeasure(kind='correlation')
        simulated_fc = connectivity_measure.fit_transform([simulated_ts])[0]
        discrepancy = np.mean((empirical_fc - simulated_fc) ** 2)
        return discrepancy

    result = dual_annealing(objective_function, bounds)
    return result.x, result.fun

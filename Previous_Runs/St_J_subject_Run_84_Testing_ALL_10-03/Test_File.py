from Full_Core import *

# Dummy Jij for testing purposes
N = 5  # Number of spins
temp_start = 0.1
temp_end = 3.5
steps_eq = 50000
steps_mc = 5000
Jij = None  # Assuming no Jij matrix for a simple test
# np.random.seed(0)
# Jij = np.random.rand(N, N)  # Random Jij matrix for testing purposes
test_metropolis_and_find_tc(N, temp_start, temp_end, steps_eq, steps_mc, Jij, mu=None)

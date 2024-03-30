from Full_Core_New import *


base_folder = "/home/brainlab-qm/Desktop/Ising_test_10_03/Mean"
base_output_folder = "/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Mean_Analysis_Other"

# np.random.seed(7) # Ensuring Reproducibility
record_and_analyze_parallel(base_folder, base_output_folder, max_workers=8)

from Full_Core import *


base_folder = "/home/brainlab-qm/Desktop/5_node_Ising/To_Analyze"
base_output_folder = "/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Run_3_2k_4k_without_mu/"

# np.random.seed(7) # Ensuring Reproducibility
record_and_analyze_parallel(base_folder, base_output_folder, max_workers=8)

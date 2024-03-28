from Full_Core import *

# base_folder = "/home/brainlab-qm/Desktop/Analysing_Ising_model/To_Analyze/"
# base_output_folder = "/home/brainlab-qm/Desktop/Analysing_Ising_model/Output/"
# read_and_process_files(base_folder,base_output_folder)
# record_and_analyze(base_folder, base_output_folder)




base_folder = "/home/brainlab-qm/Desktop/Ising_test_10_03/To_Analyze"
base_output_folder = "/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Test_1"
# np.random.seed(7) # Ensuring Reproducibility
record_and_analyze_parallel(base_folder, base_output_folder, max_workers=8)


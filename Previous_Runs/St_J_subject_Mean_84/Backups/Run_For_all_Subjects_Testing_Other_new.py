from Previous_Runs.St_J_subject_Mean_Testing_10_21.Backups.Full_Core_New import *


base_folder = "/home/brainlab-qm/Desktop/Ising_test_10_03/Mean"
base_output_root = "/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Mean_Analysis_Other"

num_runs = 5  # Specify the number of runs you want to execute

# np.random.seed(7)  # Ensuring reproducibility for all runs
if not os.path.exists(base_output_root):
    os.makedirs(base_output_root)

for run_no in range(1, num_runs + 1):

    base_output_folder = os.path.join(base_output_root, f"Run_test_{run_no}")

    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)

    print(f"Starting run {run_no}, saving output to {base_output_folder}")
    record_and_analyze_parallel(base_folder, base_output_folder, max_workers=4)
    print(f"Run {run_no} completed.")

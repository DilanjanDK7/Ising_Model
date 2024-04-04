from Full_Core import *
import warnings
warnings.filterwarnings("ignore")

base_folder = "/home/brainlab-qm/Desktop/Ising_test_10_03/Mean"
base_output_root = "/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Mean_Analysis_With_FC_both_7_2"

num_runs = 3  # Specify the number of runs you want to execute


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

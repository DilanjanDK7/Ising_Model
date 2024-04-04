from Full_Core_Seed_parallelised import *


def run_analysis(base_folder, base_output_root, num_runs=10):
    """
    Execute the analysis for a specified number of runs, saving the output in separate directories.

    Parameters:
    - base_folder: str, path to the input data directory.
    - base_output_root: str, root path for saving the analysis outputs.
    - num_runs: int, the number of analysis runs to perform.
    """
    # Uncomment the following line to ensure reproducibility across runs
    # np.random.seed(7)

    # Create the base output directory if it does not exist
    if not os.path.exists(base_output_root):
        os.makedirs(base_output_root)

    for run_no in range(1, num_runs + 1):
        base_output_folder = os.path.join(base_output_root, f"Run_test_{run_no}")

        # Create a specific output directory for this run if it does not exist
        if not os.path.exists(base_output_folder):
            os.makedirs(base_output_folder)

        print(f"Starting run {run_no}, saving output to {base_output_folder}")

        # Execute the analysis and parallel recording function
        try:
            record_and_analyze_parallel(base_folder, base_output_folder, max_workers=1)
        except Exception as e:
            print(f"An error occurred during run {run_no}: {e}")
        else:
            print(f"Run {run_no} completed.")


if __name__ == "__main__":
    base_folder = r"D:\DIlanjan\Ising_data\Mean_84"
    base_output_root = r"D:\DIlanjan\Ising_outputs_84\Seeds_test_parallel_without_mu_jij_biased"
    os.makedirs(base_output_root,exist_ok=True)
    num_runs = 10  # Specify the number of runs you want to execute

    run_analysis(base_folder, base_output_root, num_runs)

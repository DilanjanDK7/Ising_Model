from Full_Core import *
from scipy import stats
import os
import logging
import pandas as pd
from scipy.stats import pearsonr, f_oneway
from concurrent.futures import ProcessPoolExecutor # For parallel processing
import concurrent.futures as cf


def write_values_to_file(output_folder, description_array, value_array):
    """
    Writes descriptions and their corresponding values to a text file.

    Parameters:
    - output_folder: str, the path to the folder where the output file will be saved.
    - description_array: list of str, the descriptions to be written to the file.
    - value_array: list, the values corresponding to each description.

    The function saves a file named 'output.txt' in the specified output folder,
    writing each description and its corresponding value in the format 'description: value'.
    """

    # Ensure the output folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the output file path
    file_path = os.path.join(output_folder, 'output.txt')

    # Open the file for writing
    with open(file_path, 'w') as file:
        # Iterate over both arrays simultaneously
        for description, value in zip(description_array, value_array):
            # Write the description and value to the file
            file.write(f'{description}: {value}\n')

    print(f'Values have been written to {file_path}')

'''
def perform_statistical_analysis(df):
    # Example analysis: Correlation between empirical and simulated correlation
    correlations = df[['empirical_simulated', 'empirical_jij', 'simulated_jij']]
    print("Correlations among metrics:")
    print(correlations.corr())

    # ANOVA for differences between subjects in one of the metrics, e.g., empirical_simulated
    subjects = df['subject_id'].unique()
    if len(subjects) > 1:
        # Assuming there are multiple subjects to compare
        anova_data = [df[df['subject_id'] == subject]['empirical_simulated'].values for subject in subjects]
        anova_result = f_oneway(*anova_data)
        print(f"ANOVA result for differences between subjects in empirical_simulated: F={anova_result.statistic}, p={anova_result.pvalue}")
'''

def perform_statistical_analysis(df):
    # Check for the existence of required columns before proceeding
    required_columns = ['empirical_simulated', 'empirical_jij', 'simulated_jij']
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        logging.warning(f"Missing columns in DataFrame: {missing_columns}. Analysis may be limited.")
    else:
        # Proceed with analysis if all columns are present
        correlations = df[required_columns]
        print("Correlations among metrics:")
        print(correlations.corr())

    # Example of handling missing data gracefully
    if 'subject_id' in df.columns and 'empirical_simulated' in df.columns:
        subjects = df['subject_id'].unique()
        if len(subjects) > 1:
            anova_data = [df[df['subject_id'] == subject]['empirical_simulated'].values for subject in subjects]
            anova_result = f_oneway(*anova_data)
            print(f"ANOVA result for differences between subjects in empirical_simulated: F={anova_result.statistic}, p={anova_result.pvalue}")
        else:
            logging.warning("Insufficient subjects for ANOVA.")
    else:
        logging.warning("Missing 'subject_id' or 'empirical_simulated' column for ANOVA analysis.")
def save_to_csv(df, output_path):
    # Save the DataFrame to a CSV file for further analysis or record-keeping
    df.to_csv(output_path, index=False)
def create_individual_output_folders(base_output_folder, subject_id, parcellation):
    """
    Creates individualized output folders for a given subject and parcellation combination
    within a specified base output folder.

    Parameters:
    - base_output_folder: The root directory where output folders will be created.
    - subject_id: The identifier for the subject.
    - parcellation: The name of the parcellation scheme being used.

    The function constructs a path using these parameters and ensures the directory exists.
    """
    # Construct the path for the new output folder
    output_path = os.path.join(base_output_folder, subject_id, parcellation)

    # Create the directory if it does not already exist
    os.makedirs(output_path, exist_ok=True)

    # Optionally, return the path of the created folder for further use
    return output_path

def process_files(fmri_path, pet_path, jij_path,subject_id,parcellation,base_output_folder):
    """
    Placeholder function to process or perform calculations on the fmri, pet, and jij files.

    Parameters:
    - fmri_path: Path to the fmri file.
    - pet_path: Path to the pet file.
    - jij_path: Path to the jij file.

    This function should contain the specific analysis or processing steps for the provided files.
    For demonstration purposes, it just prints the paths of the files.
    """
    # Placeholder for actual processing logic
    output_path= create_individual_output_folders(base_output_folder, subject_id, parcellation)
    print(f"Processing:\nFMRI: {fmri_path}\nPET: {pet_path}\nJIJ: {jij_path}")
    time_series_path = fmri_path
    N = 5
    steps_eq = 500
    steps_mc = 3000
    Jij = np.loadtxt(jij_path, delimiter=',')
    mu = np.loadtxt(pet_path, delimiter=',')
    Jij = normalize_matrix(Jij) # Normalizing Jij
    optimised =optimize_and_simulate(time_series_path, N, steps_eq, steps_mc, output_path, Jij=Jij, mu=mu)
    description_array = ['Optimal_Temperature', 'Optimal_Alpha', 'Parcellation', 'Subject','correlations']
    value_array = [optimised[0][0], optimised[0][1], parcellation, subject_id,optimised[1]]
    write_values_to_file(output_path, description_array, value_array)

    print(" Succesfully Completed ")
    return optimised

''''
def read_and_process_files(base_folder,base_output_folder):
    """
    Read the paths of copied fmri, pet, and jij files from structured subfolders within a base folder
    and perform a calculation or analysis on them.

    Parameters:
    - base_folder: The base directory containing the structured subfolders of copied files.
    """
    # Attempt to list directories and handle possible FileNotFoundError
    try:
        subjects = os.listdir(base_folder)
    except FileNotFoundError:
        logging.error(f"The directory {base_folder} was not found.")
        return

    for subject_id in subjects:
        subject_path = os.path.join(base_folder, subject_id)

        if os.path.isdir(subject_path):
            for parcellation in os.listdir(subject_path):
                parcellation_path = os.path.join(subject_path, parcellation)

                # Initialize file paths
                fmri_path = pet_path = jij_path = None

                try:
                    for file_name in os.listdir(parcellation_path):
                        file_path = os.path.join(parcellation_path, file_name)

                        # Identify and assign the correct file path
                        if file_name.endswith('time_series.csv'):
                            fmri_path = file_path
                        elif file_name.endswith('features.txt'):
                            pet_path = file_path
                        elif file_name.endswith('Jij.csv'):
                            jij_path = file_path

                    # Check if all file paths are identified before processing
                    if fmri_path and pet_path and jij_path:
                        optimized_values = process_files(fmri_path, pet_path, jij_path,subject_id,parcellation,base_output_folder)
                    else:
                        logging.warning(f"Missing files for {subject_id} in parcellation {parcellation}. Skipping...")

                except Exception as e:
                    logging.error(f"Error accessing files in {parcellation_path}: {e}")
'''
'''
def read_and_process_files(base_folder, base_output_folder):
    """
    Extended function to read files, perform calculations, and tabulate results.
    """
    results = []  # List to store results for tabulation

    try:
        subjects = os.listdir(base_folder)
    except FileNotFoundError:
        logging.error(f"The directory {base_folder} was not found.")
        return

    for subject_id in subjects:
        subject_path = os.path.join(base_folder, subject_id)
        if os.path.isdir(subject_path):
            for parcellation in os.listdir(subject_path):
                fmri_path = pet_path = jij_path = None
                parcellation_path = os.path.join(subject_path, parcellation)
                try:
                    for file_name in os.listdir(parcellation_path):
                        file_path = os.path.join(parcellation_path, file_name)
                        if file_name.endswith('time_series.csv'):
                            fmri_path = file_path
                        elif file_name.endswith('features.txt'):
                            pet_path = file_path
                        elif file_name.endswith('Jij.csv'):
                            jij_path = file_path

                    if fmri_path and pet_path and jij_path:
                        optimized_values = process_files(fmri_path, pet_path, jij_path, subject_id, parcellation, base_output_folder)
                        # Record optimized values with subject and parcellation identifiers
                        results.append((subject_id, parcellation, optimized_values))
                except Exception as e:
                    logging.error(f"Error accessing files in {parcellation_path}: {e}")

    # Create a DataFrame from the results
    df = pd.DataFrame(results, columns=['SubjectID', 'Parcellation', 'OptimizedValues'])

    # Perform statistical analysis here on 'df'
    # Example: Mean and standard deviation of 'OptimizedValues' for each subject
    # and across all subjects

    return df
'''
# Example usage:
# base_folder = "/media/ubuntu/Elements/Dilanjan_Run/Ising_model/5x5_new/"
# read_and_process_files(base_folder)
def read_and_process_files(subject_path, parcellation, base_output_folder):
    """
    Function to read files for a given subject and parcellation, perform calculations, and tabulate results.
    """
    results = []  # List to store results for tabulation
    parcellation_path = os.path.join(subject_path, parcellation)
    fmri_path = pet_path = jij_path = None
    try:
        for file_name in os.listdir(parcellation_path):
            file_path = os.path.join(parcellation_path, file_name)
            if file_name.endswith('time_series.csv'):
                fmri_path = file_path
            elif file_name.endswith('features.txt'):
                pet_path = file_path
            elif file_name.endswith('Jij.csv'):
                jij_path = file_path

        if fmri_path and pet_path and jij_path:
            # Process the files for this specific subject and parcellation
            subject_id = os.path.basename(subject_path)
            optimized_values = process_files(fmri_path, pet_path, jij_path, subject_id, parcellation, base_output_folder)
            # Record optimized values with subject and parcellation identifiers
            results.append((subject_id, parcellation, optimized_values))
    except Exception as e:
        logging.error(f"Error accessing files in {parcellation_path}: {e}")
        return None

    # Create a DataFrame from the results, if any
    if results:
        df = pd.DataFrame(results, columns=['SubjectID', 'Parcellation', 'OptimizedValues'])
        return df
    else:
        return None

'''
def record_and_analyze(base_folder, base_output_folder):
    # Container for all subjects' optimized values and metrics
    all_data = []

    subjects = os.listdir(base_folder)
    for subject_id in subjects:
        subject_path = os.path.join(base_folder, subject_id)
        if os.path.isdir(subject_path):
            for parcellation in os.listdir(subject_path):
                parcellation_path = os.path.join(subject_path, parcellation)
                # Assuming process_files function returns a structure as in the provided example
                optimized_values = read_and_process_files(parcellation_path, base_output_folder)
                if optimized_values:
                    # Append a dictionary containing the subject ID, parcellation, and the optimized values
                    all_data.append({
                        "subject_id": subject_id,
                        "parcellation": parcellation,
                        **optimized_values[1]  # Unpacking the dictionary of metrics
                    })

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_data)

    # Perform statistical analysis
    perform_statistical_analysis(df)
'''
def record_and_analyze(base_folder, base_output_folder):
    all_data = []  # Container for all subjects' optimized values and metrics

    try:
        subjects = os.listdir(base_folder)
    except FileNotFoundError:
        logging.error(f"The directory {base_folder} was not found.")
        return

    for subject_id in subjects:
        subject_path = os.path.join(base_folder, subject_id)
        if os.path.isdir(subject_path):
            for parcellation in os.listdir(subject_path):
                # Process each subject and parcellation combination
                df = read_and_process_files(subject_path, parcellation, base_output_folder)
                if df is not None:
                    all_data.extend(df.to_dict('records'))

    # Convert to DataFrame for easier manipulation
    if all_data:
        df = pd.DataFrame(all_data)
        # Perform statistical analysis
        perform_statistical_analysis(df)  # Placeholder for actual analysis function
        return df
    else:
        return None


def record_and_analyze_parallel(base_folder, base_output_folder, max_workers=4):
    all_data = []  # Container for all subjects' optimized values and metrics

    try:
        subjects = os.listdir(base_folder)
    except FileNotFoundError:
        logging.error(f"The directory {base_folder} was not found.")
        return

    # Create a process pool executor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for subject_id in subjects:
            subject_path = os.path.join(base_folder, subject_id)
            if os.path.isdir(subject_path):
                for parcellation in os.listdir(subject_path):
                    # Submit each subject and parcellation combination as a separate task
                    futures.append(
                        executor.submit(read_and_process_files, subject_path, parcellation, base_output_folder))

        # Wait for all submitted tasks to complete and collect their results
        for future in cf.as_completed(futures):
            df = future.result()
            if df is not None:
                all_data.extend(df.to_dict('records'))

    # Convert to DataFrame for easier manipulation
    if all_data:
        df = pd.DataFrame(all_data)
        # Perform statistical analysis
        perform_statistical_analysis(df)  # Placeholder for actual analysis function
        return df
    else:
        return None


# base_folder = "/home/brainlab-qm/Desktop/Analysing_Ising_model/To_Analyze/"
# base_output_folder = "/home/brainlab-qm/Desktop/Analysing_Ising_model/Output/"
# read_and_process_files(base_folder,base_output_folder)
# record_and_analyze(base_folder, base_output_folder)

base_folder = "/home/brainlab-qm/Desktop/5_node_Ising/To_Analyze"
base_output_folder = "/home/brainlab-qm/Desktop/5_node_Ising/Output_with_mu_optimize_both_remove early_steps"


np.random.seed(42) # Ensuring Reproducibility
record_and_analyze_parallel(base_folder, base_output_folder, max_workers=8)

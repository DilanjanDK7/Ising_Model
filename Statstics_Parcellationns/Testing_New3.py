import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests



def load_subject_files(base_path):
    """
    Load specific .npy files for all subjects and parcellations into Pandas DataFrames.

    Parameters:
    - base_path: String, the base directory path containing all subject data.

    Returns:
    A dictionary with keys as file types and values as a list of tuples. Each tuple contains subject ID,
    parcellation, and the loaded data as a DataFrame.
    """
    data_types = ['time_series_empirical.npy', 'time_series_optimized.npy',
                  'Simulated_fc_matrix_optimized.npy', 'Empirical_fc_matrix_optimized.npy', 'Jij_optimized.npy']
    loaded_data = {data_type: [] for data_type in data_types}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file in data_types:
                file_path = os.path.join(root, file)
                path_parts = root.split(os.sep)
                subject_id_index = -2  # Adjust based on your file structure
                parcellation_index = -1  # Adjust based on your file structure
                subject_id = path_parts[subject_id_index]
                parcellation = path_parts[parcellation_index]
                data = np.load(file_path)
                df = pd.DataFrame(data)
                loaded_data[file].append((subject_id, parcellation, df))

    return loaded_data

def compute_fc_correlations(loaded_data):
    """
    Compute Pearson correlation coefficients for the upper triangle (excluding the diagonal)
    between empirical and simulated, empirical and Jij, simulated and Jij FC matrices.

    Parameters:
    - loaded_data: The loaded data dictionary from `load_subject_files`.

    Returns:
    A DataFrame containing the subject ID, parcellation, and the correlation coefficient for each combination.
    """
    correlations = []
    comparison_pairs = [
        ('Empirical_fc_matrix_optimized.npy', 'Simulated_fc_matrix_optimized.npy'),
        ('Empirical_fc_matrix_optimized.npy', 'Jij_optimized.npy'),
        ('Simulated_fc_matrix_optimized.npy', 'Jij_optimized.npy')
    ]

    for pair in comparison_pairs:
        for empirical_data, comparison_data in zip(loaded_data[pair[0]], loaded_data[pair[1]]):
            subject_id, parcellation, empirical_df = empirical_data
            _, _, comparison_df = comparison_data
            upper_tri_index = np.triu_indices_from(empirical_df, k=1)
            empirical_upper_tri = empirical_df.values[upper_tri_index]
            comparison_upper_tri = comparison_df.values[upper_tri_index]
            correlation = np.corrcoef(empirical_upper_tri, comparison_upper_tri)[0, 1]
            correlations.append((subject_id, parcellation, pair[0], pair[1], correlation))

    return pd.DataFrame(correlations, columns=['Subject ID', 'Parcellation', 'Data Type 1', 'Data Type 2', 'Correlation'])

def perform_group_analysis(correlation_df, output_folder):
    """
    Perform group analysis on the correlation coefficients for each pair of conditions
    (Empirical vs. Simulated, Empirical vs. Jij, Simulated vs. Jij), computing descriptive statistics,
    visualizations, and logging data modifications.

    Parameters:
    - correlation_df: DataFrame containing the correlation coefficients.
    - output_folder: The directory path to save plots, summary data, and logs.
    """

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Log file setup
    log_file_path = os.path.join(output_folder, 'analysis_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Analysis Log\n")
        log_file.write("=========================\n")

        # Handle NaN values and constant columns before analysis
        correlation_df_clean = correlation_df.dropna(subset=['Correlation'])  # Ensure to drop NaNs only in 'Correlation'

        # Identify and exclude non-numeric columns for variance filtering
        numeric_cols = correlation_df_clean.select_dtypes(include=[np.number]).columns.tolist()
        # Now apply variance threshold only to numeric columns
        variance_threshold = 0.0
        non_constant_numeric_cols = correlation_df_clean[numeric_cols].var() > variance_threshold
        # Ensure to keep the non-numeric columns ('Subject ID', 'Parcellation') in your DataFrame
        cols_to_keep = ['Subject ID', 'Parcellation','Data Type 1', 'Data Type 2'] + non_constant_numeric_cols[non_constant_numeric_cols].index.tolist()
        correlation_df_clean = correlation_df_clean[cols_to_keep]

        # Descriptive statistics and visualizations for each comparison
        for types in correlation_df_clean[['Data Type 1', 'Data Type 2']].drop_duplicates().values:
            data_type_1, data_type_2 = types
            subset = correlation_df_clean[(correlation_df_clean['Data Type 1'] == data_type_1) & (correlation_df_clean['Data Type 2'] == data_type_2)]
            descriptive_stats = subset['Correlation'].describe()

            # Log descriptive statistics
            log_file.write(f"\nDescriptive statistics for {data_type_1} vs. {data_type_2}:\n")
            log_file.write(descriptive_stats.to_string())
            log_file.write("\n\n")

            # Visualization: Histogram of correlation coefficients
            plt.figure(figsize=(12, 8))
            sns.histplot(subset['Correlation'], kde=True, stat="density", linewidth=0)
            plt.title(f'Correlation Coefficients Distribution: {data_type_1} vs. {data_type_2}')
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Density')
            plt.savefig(os.path.join(output_folder, f'hist_{data_type_1}_vs_{data_type_2}.png'))
            plt.close()

    print(f"Analysis completed. Results and plots saved in {output_folder}")

# Ensure base_path is defined and points to your data directory before calling load_subject_files.
# loaded_data = load_subject_files(base_path)

# Once loaded_data is obtained, calculate correlations.
# correlation_df = compute_fc_correlations(loaded_data)

# Ensure output_folder is defined and points to your desired output directory before calling perform_group_analysis.
# perform_group_analysis(correlation_df, output_folder)

loaded_data = load_subject_files('/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Run_1')
coorelation_df = compute_fc_correlations(loaded_data)
perform_group_analysis(coorelation_df, '/home/brainlab-qm/Desktop/Ising_test_10_03/Group_Analysis/Run_1_Test2')

print(" ")
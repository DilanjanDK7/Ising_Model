import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests




# def load_subject_files(base_path):
#     """
#     Load specific .npy files for all subjects and parcellations into Pandas DataFrames.
#
#     Parameters:
#     - base_path: String, the base directory path containing all subject data.
#
#     Returns:
#     A dictionary with keys as file types and values as a list of tuples. Each tuple contains subject ID,
#     parcellation, and the loaded data as a DataFrame.
#     """
#     data_types = ['time_series_empirical.npy', 'time_series_optimized.npy',
#                   'Simulated_fc_matrix_optimized.npy', 'Empirical_fc_matrix_optimized.npy']
#     loaded_data = {data_type: [] for data_type in data_types}
#
#     for root, dirs, files in os.walk(base_path):
#         for file in files:
#             if file in data_types:
#                 file_path = os.path.join(root, file)
#                 subject_id = root.split(os.sep)[-2]  # Assuming subject ID is Two levels up from file
#                 parcellation = root.split(os.sep)[-1]  # Assuming parcellation is One level up from file
#                 data = np.load(file_path)
#                 df = pd.DataFrame(data)
#                 loaded_data[file].append((subject_id, parcellation, df))
#
#     return loaded_data

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
                  'Simulated_fc_matrix_optimized.npy', 'Empirical_fc_matrix_optimized.npy']
    loaded_data = {data_type: [] for data_type in data_types}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file in data_types:
                file_path = os.path.join(root, file)
                path_parts = root.split(os.sep)
                # Assuming the specific structure for parsing subject ID and parcellation
                subject_id_index = -2  # Adjusted based on your file structure
                parcellation_index = -1  # Adjusted based on your file structure
                subject_id = path_parts[subject_id_index]
                parcellation = path_parts[parcellation_index]
                data = np.load(file_path)
                df = pd.DataFrame(data)
                loaded_data[file].append((subject_id, parcellation, df))

    return loaded_data
def compute_fc_correlations(loaded_data):
    """
    Compute Pearson correlation coefficients between simulated and empirical FC matrices.

    Parameters:
    - loaded_data: The loaded data dictionary from `load_subject_files`.

    Returns:
    A DataFrame containing the subject ID, parcellation, and the correlation coefficient.
    """
    correlations = []
    for empirical, simulated in zip(loaded_data['Empirical_fc_matrix_optimized.npy'],
                                    loaded_data['Simulated_fc_matrix_optimized.npy']):
        subject_id, parcellation, empirical_df = empirical
        _, _, simulated_df = simulated
        correlation_matrix = empirical_df.corrwith(simulated_df, axis=0)
        mean_correlation = correlation_matrix.mean()
        correlations.append((subject_id, parcellation, mean_correlation))

    return pd.DataFrame(correlations, columns=['Subject ID', 'Parcellation', 'Correlation'])


def perform_group_analysis(correlation_df, output_folder):
    """
    Enhanced group analysis on the correlation coefficients with multi-level DataFrame,
    including ANOVA, t-tests, network correlations, deviations, violin plots, and more.

    Parameters:
    - correlation_df: DataFrame with multi-level index (Subject ID, Parcellation) containing the correlation coefficients.
    - output_folder: The directory path to save plots and summary data.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    correlation_df_reset = correlation_df.reset_index()

    descriptive_stats = correlation_df_reset.groupby('Parcellation')['Correlation'].describe()
    print(descriptive_stats)

    # ANOVA
    groups = correlation_df_reset.groupby('Parcellation')['Correlation'].apply(list).values
    f_value, p_value = stats.f_oneway(*groups)
    print(f"\nANOVA test across all parcellations: F = {f_value}, p = {p_value}")

    # Pairwise t-tests with correction for multiple comparisons (Welch's t-test)
    parcellations = correlation_df_reset['Parcellation'].unique()
    pairwise_tests = []
    if len(parcellations) > 1:
        for i in range(len(parcellations)):
            for j in range(i+1, len(parcellations)):
                data_i = correlation_df_reset[correlation_df_reset['Parcellation'] == parcellations[i]]['Correlation']
                data_j = correlation_df_reset[correlation_df_reset['Parcellation'] == parcellations[j]]['Correlation']
                t_stat, p_val = stats.ttest_ind(data_i, data_j, equal_var=False)
                pairwise_tests.append((parcellations[i], parcellations[j], t_stat, p_val))

    # Correcting for multiple comparisons (e.g., FDR, Bonferroni)
    p_vals = [x[3] for x in pairwise_tests]
    reject, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
    corrected_results = zip(pairwise_tests, pvals_corrected, reject)
    # Assuming pairwise_tests is a list of tuples, where each tuple is like (parcellation_i, parcellation_j, t_stat, p_val)
    # and you've run the multipletests function to get pvals_corrected and reject arrays

    for pair, corrected_pval, is_rejected in zip(pairwise_tests, pvals_corrected, reject):
        parcellation_i, parcellation_j, t_stat, p_val = pair  # Unpack the details from each pairwise test
        print(f"{parcellation_i} vs. {parcellation_j}: corrected p = {corrected_pval}, reject null: {is_rejected}")

    # for (pair, _, _), corrected_pval, reject in corrected_results:
    #     print(f"{pair[0]} vs. {pair[1]}: corrected p = {corrected_pval}, reject null: {reject}")

    # Save descriptive statistics and pairwise comparisons
    descriptive_stats.to_csv(os.path.join(output_folder, 'descriptive_statistics.csv'))

    # Plotting mean correlations with error bars
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Parcellation', y='Correlation', data=correlation_df_reset, capsize=.2)
    plt.title('Mean Correlation by Parcellation with Error Bars')
    plt.xlabel('Parcellation')
    plt.ylabel('Mean Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mean_correlation_by_parcellation_with_error.png'))
    plt.close()

    # Violin plot to show distributions, including mean and std
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Parcellation', y='Correlation', data=correlation_df_reset, inner="quartile")
    plt.title('Distribution of Correlations by Parcellation')
    plt.xlabel('Parcellation')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'correlation_distributions_by_parcellation.png'))
    plt.close()

    # Other analyses placeholders
    # e.g., network analysis, modularity, centrality, etc.

    print(f"Enhanced analysis completed. Results and plots saved in {output_folder}")
# def perform_group_analysis(correlation_df, output_folder):
#     """
#     Perform group analysis on the correlation coefficients with multi-level DataFrame,
#     including ANOVA, t-tests, network correlations, deviations, and visualizations.
#
#     Parameters:
#     - correlation_df: DataFrame with multi-level index (Subject ID, Parcellation) containing the correlation coefficients.
#     - output_folder: The directory path to save plots and summary data.
#
#     Saves plots and a summary DataFrame to the specified output folder.
#     """
#     # Ensure the output folder exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Reset index for easier manipulation
#     correlation_df_reset = correlation_df.reset_index()
#
#     # Descriptive statistics
#     descriptive_stats = correlation_df_reset.groupby('Parcellation')['Correlation'].describe()
#     print(descriptive_stats)
#
#     # ANOVA to test for differences across all parcellations
#     groups = correlation_df_reset.groupby('Parcellation')['Correlation'].apply(list).values
#     f_value, p_value = stats.f_oneway(*groups)
#     print(f"\nANOVA test across all parcellations: F = {f_value}, p = {p_value}")
#
#     # Pairwise t-tests between parcellations
#     parcellations = correlation_df_reset['Parcellation'].unique()
#     if len(parcellations) > 1:
#         print("\nPairwise t-tests between parcellations:")
#         for i in range(len(parcellations)):
#             for j in range(i + 1, len(parcellations)):
#                 data_i = correlation_df_reset[correlation_df_reset['Parcellation'] == parcellations[i]]['Correlation']
#                 data_j = correlation_df_reset[correlation_df_reset['Parcellation'] == parcellations[j]]['Correlation']
#                 t_stat, p_val = stats.ttest_ind(data_i, data_j, equal_var=False)  # Welch's t-test
#                 print(f"  {parcellations[i]} vs. {parcellations[j]}: t = {t_stat}, p = {p_val}")
#
#     # Save descriptive statistics to CSV
#     descriptive_stats.to_csv(os.path.join(output_folder, 'descriptive_statistics.csv'))
#
#     # Group-Level Analysis (Placeholder for actual analysis)
#     # Example: Calculate mean correlation for each parcellation and plot
#     mean_correlations = correlation_df_reset.groupby('Parcellation')['Correlation'].mean()
#     plt.figure(figsize=(10, 6))
#     mean_correlations.plot(kind='bar')
#     plt.title('Mean Correlation by Parcellation')
#     plt.xlabel('Parcellation')
#     plt.ylabel('Mean Correlation')
#     plt.savefig(os.path.join(output_folder, 'mean_correlation_by_parcellation.png'))
#     plt.close()
#
#     # Add additional analyses as required (e.g., network correlations, deviations)
#     # This step depends on the specific analyses you wish to perform and the data available.
#
#     print(f"Analysis completed. Results and plots saved in {output_folder}")


loaded_data = load_subject_files('/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Run_1')
coorelation_df = compute_fc_correlations(loaded_data)
perform_group_analysis(coorelation_df, '/home/brainlab-qm/Desktop/Ising_test_10_03/Group_Analysis/Run_1_Test')

print(" ")
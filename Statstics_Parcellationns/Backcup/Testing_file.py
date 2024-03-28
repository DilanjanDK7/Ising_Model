import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

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
                subject_id = root.split(os.sep)[-3]  # Assuming subject ID is three levels up from file
                parcellation = root.split(os.sep)[-2]  # Assuming parcellation is two levels up from file
                data = np.load(file_path)
                df = pd.DataFrame(data)
                loaded_data[file].append((subject_id, parcellation, df))

    return loaded_data

def compute_fc_correlations(loaded_data, network_name):
    """
    Compute Pearson correlation coefficients for a specified network between simulated and empirical FC matrices,
    considering variations between subjects.

    Parameters:
    - loaded_data: The loaded data dictionary from `load_subject_files`.
    - network_name: String, the name of the network for which correlations are to be calculated.

    Returns:
    A DataFrame containing the subject ID, parcellation, network name, and the correlation coefficient.
    """
    correlations = []
    for empirical, simulated in zip(loaded_data['Empirical_fc_matrix_optimized.npy'],
                                     loaded_data['Simulated_fc_matrix_optimized.npy']):
        subject_id, parcellation, empirical_df = empirical
        _, _, simulated_df = simulated
        # Assuming the network data is a subset of columns, identified by network_name
        network_columns = [col for col in empirical_df.columns if network_name in col]  # Adjust as necessary
        correlation_matrix = empirical_df[network_columns].corrwith(simulated_df[network_columns], axis=0)
        mean_correlation = correlation_matrix.mean()
        correlations.append((subject_id, parcellation, network_name, mean_correlation))

    return pd.DataFrame(correlations, columns=['Subject ID', 'Parcellation', 'Network', 'Correlation'])

def perform_group_analysis(correlation_df, output_folder):
    """
    Perform group analysis on the correlation coefficients with multi-level DataFrame,
    including ANOVA, t-tests, network correlations, deviations, and visualizations,
    considering variations between subjects for a given network.

    Parameters:
    - correlation_df: DataFrame with multi-level index (Subject ID, Parcellation, Network) containing the correlation coefficients.
    - output_folder: The directory path to save plots and summary data.

    Saves plots and a summary DataFrame to the specified output folder.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Reset index for easier manipulation
    correlation_df_reset = correlation_df.reset_index()

    # Descriptive statistics
    descriptive_stats = correlation_df_reset.groupby(['Parcellation', 'Network'])['Correlation'].describe()
    print(descriptive_stats)

    # ANOVA to test for differences across all parcellations and networks
    for network in correlation_df_reset['Network'].unique():
        groups = correlation_df_reset[correlation_df_reset['Network'] == network].groupby('Parcellation')['Correlation'].apply(list).values
        f_value, p_value = stats.f_oneway(*groups)
        print(f"\nANOVA test for {network} across all parcellations: F = {f_value}, p = {p_value}")

    # Pairwise t-tests between parcellations for each network
    print("\nPairwise t-tests between parcellations for each network:")
    for network in correlation_df_reset['Network'].unique():
        print(f"\nNetwork: {network}")
        parcellations = correlation_df_reset[correlation_df_reset['Network'] == network]['Parcellation'].unique()
        for i in range(len(parcellations)):
            for j in range(i + 1, len(parcellations)):
                data_i = correlation_df_reset[(correlation_df_reset['Parcellation'] == parcellations[i]) & (correlation_df_reset['Network'] == network)]['Correlation']
                data_j = correlation_df_reset[(correlation_df_reset['Parcellation'] == parcellations[j]) & (correlation_df_reset['Network'] == network)]['Correlation']
                t_stat, p_val = stats.ttest_ind(data_i, data_j, equal_var=False)  # Welch's t-test
                print(f"  {parcellations[i]} vs. {parcellations[j]}: t = {t_stat}, p = {p_val}")

    # Save descriptive statistics to CSV
    descriptive_stats.to_csv(os.path.join(output_folder, 'descriptive_statistics.csv'))

    # Plotting mean correlations by parcellation and network
    plt.figure(figsize=(10, 6))
    for network in correlation_df_reset['Network'].unique():
        mean_correlations = correlation_df_reset[correlation_df_reset['Network'] == network].groupby('Parcellation')['Correlation'].mean()
        mean_correlations.plot(kind='bar', label=network, figsize=(10,6))
    plt.title('Mean Correlation by Parcellation and Network')
    plt.xlabel('Parcellation')
    plt.ylabel('Mean Correlation')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'mean_correlation_by_parcellation_and_network.png'))
    plt.close()

    print(f"Analysis completed. Results and plots saved in {output_folder}")

# Note: Before running these functions, ensure you have your environment set up with the necessary libraries (e.g., NumPy, Pandas, SciPy, Matplotlib)
# and that your data is organized appropriately for this code to function correctly.

loaded_data = load_subject_files('/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Run_1')
coorelation_df = compute_fc_correlations(loaded_data)
perform_group_analysis(coorelation_df, '/home/brainlab-qm/Desktop/Ising_test_10_03/Group_Analysis/Run_1_Test')

print(" ")
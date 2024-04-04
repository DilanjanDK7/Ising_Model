import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

def load_subject_files(base_path):
    """
    Load specific .npy files for all subjects and parcellations into Pandas DataFrames.
    """
    data_types = ['time_series_empirical.npy', 'time_series_optimized.npy',
                  'Simulated_fc_matrix_optimized.npy', 'Empirical_fc_matrix_optimized.npy', 'Jij_optimized.npy']
    loaded_data = {data_type: [] for data_type in data_types}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file in data_types:
                file_path = os.path.join(root, file)
                path_parts = root.split(os.sep)
                subject_id = path_parts[-2]  # Assuming this is the structure
                parcellation = path_parts[-1]  # Assuming this is the structure
                data = np.load(file_path)
                df = pd.DataFrame(data)
                loaded_data[file].append((subject_id, parcellation, df))

    return loaded_data

def perform_group_analysis(correlation_df, output_folder):
    """
    Perform group analysis including descriptive statistics, ANOVA, pairwise t-tests with corrections,
    and visualizations, focusing on mean correlations and distributions by parcellation and comparison pair.
    Separate files and plots are generated for each comparison pair.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract unique pairs from the correlation_df
    comparison_pairs = correlation_df[['Data Type 1', 'Data Type 2']].drop_duplicates().values.tolist()

    for comparison_pair in comparison_pairs:
        # Filter dataframe for current comparison pair
        pair_df = correlation_df[(correlation_df['Data Type 1'] == comparison_pair[0]) &
                                 (correlation_df['Data Type 2'] == comparison_pair[1])]

        # Descriptive Statistics for each pair
        descriptive_stats = pair_df.groupby(['Parcellation', 'Data Type 1', 'Data Type 2'])['Correlation'].describe()
        descriptive_filename = f'descriptive_stats_{comparison_pair[0]}_vs_{comparison_pair[1]}.csv'
        descriptive_stats.to_csv(os.path.join(output_folder, descriptive_filename))

        # ANOVA, Pairwise tests, and Violin plots for each comparison pair
        anova_analysis(pair_df, comparison_pair, output_folder)
        perform_pairwise_tests(pair_df, comparison_pair, output_folder)
        generate_violin_plot(pair_df, comparison_pair, output_folder)  # Updated to include comparison pair in the plot

    print("Group analysis completed. Results and plots are saved in", output_folder)


def anova_analysis(df, comparison_type, output_folder):
    """
    Perform ANOVA across parcellations for a specific comparison type.
    """
    groups = df.groupby('Parcellation')['Correlation'].apply(list).values
    f_value, p_value = stats.f_oneway(*groups)
    with open(os.path.join(output_folder, f'ANOVA_{comparison_type[0]}_vs_{comparison_type[1]}.txt'), 'w') as f:
        f.write(f"ANOVA test for {comparison_type[0]} vs {comparison_type[1]}: F = {f_value}, p = {p_value}\n")


def generate_violin_plot(df, comparison_type, output_folder):
    """
    Generate violin plots for a specific comparison type.
    """
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Parcellation', y='Correlation', data=df, inner="quartile")
    plt.title(f'Distribution of Correlations: {comparison_type[0]} vs {comparison_type[1]}')
    plt.xlabel('Parcellation')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'violin_plot_{comparison_type[0]}_vs_{comparison_type[1]}.png'))
    plt.close()


def perform_pairwise_tests(df, comparison_type, output_folder):
    """
    Conduct pairwise t-tests with corrections for multiple comparisons across parcellations for a specific comparison type.
    """
    parcellations = df['Parcellation'].unique()
    pairwise_tests = []
    if len(parcellations) > 1:
        for i in range(len(parcellations)):
            for j in range(i + 1, len(parcellations)):
                data_i = df[df['Parcellation'] == parcellations[i]]['Correlation']
                data_j = df[df['Parcellation'] == parcellations[j]]['Correlation']
                t_stat, p_val = stats.ttest_ind(data_i, data_j, equal_var=False)
                pairwise_tests.append((parcellations[i], parcellations[j], t_stat, p_val))

    # Correcting for multiple comparisons
    p_vals = [x[3] for x in pairwise_tests]
    reject, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

    results_str = ""
    for i, (pair, corrected_pval, is_rejected) in enumerate(zip(pairwise_tests, pvals_corrected, reject)):
        parcellation_i, parcellation_j, t_stat, p_val = pair
        results_str += f"{parcellation_i} vs {parcellation_j}: corrected p = {corrected_pval}, reject null: {is_rejected}\n"

    with open(os.path.join(output_folder, f'pairwise_t_tests_{comparison_type[0]}_vs_{comparison_type[1]}.txt'),
              'w') as f:
        f.write(results_str)

#
def compute_fc_correlations(loaded_data):
    """
    Compute correlations for the upper triangle (excluding the diagonal) between FC matrices and return a dataframe.
    """
    # Initialize list to store correlation results
    correlations = []

    # Define comparison pairs for clarity
    comparison_pairs = [
        ('Empirical_fc_matrix_optimized.npy', 'Simulated_fc_matrix_optimized.npy'),
        ('Empirical_fc_matrix_optimized.npy', 'Jij_optimized.npy'),
        ('Simulated_fc_matrix_optimized.npy', 'Jij_optimized.npy')
    ]

    # Loop through each pair for comparison
    for pair in comparison_pairs:
        for empirical_data, comparison_data in zip(loaded_data[pair[0]], loaded_data[pair[1]]):
            # Unpack the empirical and comparison data
            subject_id, parcellation, empirical_df = empirical_data
            _, _, comparison_df = comparison_data

            # Calculate the indices for the upper triangle excluding the diagonal
            upper_tri_index = np.triu_indices_from(empirical_df, k=1)

            # Extract the upper triangle values from each matrix
            empirical_upper_tri = empirical_df.values[upper_tri_index]
            comparison_upper_tri = comparison_df.values[upper_tri_index]

            # Compute the Pearson correlation coefficient between the upper triangle values
            correlation = np.corrcoef(empirical_upper_tri, comparison_upper_tri)[0, 1]

            # Store the results
            correlations.append({
                'Subject ID': subject_id,
                'Parcellation': parcellation,
                'Data Type 1': pair[0],
                'Data Type 2': pair[1],
                'Correlation': correlation
            })

    # Convert the list of correlation results into a DataFrame for easy access
    correlations_df = pd.DataFrame(correlations)

    # Optionally, you can sort the dataframe for better readability or analysis
    correlations_df.sort_values(by=['Subject ID', 'Parcellation', 'Data Type 1', 'Data Type 2'], inplace=True)

    return correlations_df


# def compute_fc_correlations(loaded_data):
#     correlations = []
#     comparison_pairs = [
#         ('Empirical_fc_matrix_optimized.npy', 'Simulated_fc_matrix_optimized.npy'),
#         ('Empirical_fc_matrix_optimized.npy', 'Jij_optimized.npy'),
#         ('Simulated_fc_matrix_optimized.npy', 'Jij_optimized.npy')
#     ]
#
#     for pair in comparison_pairs:
#         for empirical_data, comparison_data in zip(loaded_data[pair[0]], loaded_data[pair[1]]):
#             subject_id, parcellation, empirical_df = empirical_data
#             _, _, comparison_df = comparison_data
#             upper_tri_index = np.triu_indices_from(empirical_df, k=1)
#             empirical_upper_tri = empirical_df.values[upper_tri_index]
#             comparison_upper_tri = comparison_df.values[upper_tri_index]
#
#             # Now calculating correlation and taking its absolute value
#             correlation = np.abs(np.corrcoef(empirical_upper_tri, comparison_upper_tri)[0, 1])
#
#             correlations.append({
#                 'Subject ID': subject_id,
#                 'Parcellation': parcellation,
#                 'Data Type 1': pair[0],
#                 'Data Type 2': pair[1],
#                 'Correlation': correlation
#             })
#
#     correlations_df = pd.DataFrame(correlations)
#     correlations_df.sort_values(by=['Subject ID', 'Parcellation', 'Data Type 1', 'Data Type 2'], inplace=True)
#
#     return correlations_df


loaded_data = load_subject_files('/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Run_3_2.5k_2.5k_testing_without_mu')
coorelation_df = compute_fc_correlations(loaded_data)
perform_group_analysis(coorelation_df, '/home/brainlab-qm/Desktop/Ising_test_10_03/Group_Analysis/Run_3_2.5k_2.5k_testing_without_mu')

print(" ")
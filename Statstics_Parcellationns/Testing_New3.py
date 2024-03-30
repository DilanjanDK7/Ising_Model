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

def compute_fc_correlations(loaded_data):
    """
    Compute correlations for the upper triangle (excluding the diagonal) between FC matrices.
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
    Perform group analysis including descriptive statistics, ANOVA, pairwise t-tests with corrections,
    and visualizations, focusing on mean correlations and distributions by parcellation and comparison type.
    Separate violin plots are generated for each comparison type.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    comparison_types = correlation_df[['Data Type 1', 'Data Type 2']].drop_duplicates().values.tolist()

    for comparison_type in comparison_types:
        comp_df = correlation_df[(correlation_df['Data Type 1'] == comparison_type[0]) & (correlation_df['Data Type 2'] == comparison_type[1])]

        # Descriptive Statistics
        descriptive_stats = comp_df.groupby('Parcellation')['Correlation'].describe()
        descriptive_stats.to_csv(os.path.join(output_folder, f'descriptive_statistics_{comparison_type[0]}_vs_{comparison_type[1]}.csv'))

        # ANOVA
        groups = comp_df.groupby('Parcellation')['Correlation'].apply(list).values
        f_value, p_value = stats.f_oneway(*groups)
        print(f"\nANOVA test for {comparison_type[0]} vs {comparison_type[1]}: F = {f_value}, p = {p_value}")

        # Pairwise t-tests with correction for multiple comparisons (Welch's t-test)
        perform_pairwise_tests(comp_df, comparison_type, output_folder)

        # Violin plot for each comparison type
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='Parcellation', y='Correlation', data=comp_df, inner="quartile")
        plt.title(f'Distribution of Correlations: {comparison_type[0]} vs {comparison_type[1]}')
        plt.xlabel('Parcellation')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'correlation_distributions_{comparison_type[0]}_vs_{comparison_type[1]}.png'))
        plt.close()

    print("Enhanced analysis completed. Results and plots saved in", output_folder)

def perform_pairwise_tests(df, comparison_type, output_folder):
    """
    Conduct pairwise t-tests with corrections for multiple comparisons across parcellations.
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
    for i, (pair, corrected_pval, is_rejected) in enumerate(zip(pairwise_tests, pvals_corrected, reject)):
        parcellation_i, parcellation_j, t_stat, p_val = pair
        print(f"{comparison_type[0]} vs {comparison_type[1]}, {parcellation_i} vs {parcellation_j}: corrected p = {corrected_pval}, reject null: {is_rejected}")


loaded_data = load_subject_files('/home/brainlab-qm/Desktop/Ising_test_10_03/Output/Run_2')
coorelation_df = compute_fc_correlations(loaded_data)
perform_group_analysis(coorelation_df, '/home/brainlab-qm/Desktop/Ising_test_10_03/Group_Analysis/Run_2_Test1')

print(" ")
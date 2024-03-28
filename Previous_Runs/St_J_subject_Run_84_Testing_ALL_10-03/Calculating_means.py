import os
import numpy as np
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt
import seaborn as sns


def save_matrix_as_image(matrix, title, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def save_matrix_as_csv(matrix, output_path):
    np.savetxt(output_path, matrix, delimiter=',')


def calculate_and_save_mean_matrices_within_subjects(base_folder, output_folder):
    subjects_data = {}

    subjects = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    for subject in subjects:
        atlas_path = os.path.join(base_folder, subject, "atlas_NMI_2mm.nii")
        if not os.path.exists(atlas_path):
            print(f"Atlas folder for {subject} not found.")
            continue

        jij_files, pet_files, ts_files = [], [], []

        for file in os.listdir(atlas_path):
            if file.endswith('Jij.csv'):
                jij_files.append(os.path.join(atlas_path, file))
            elif file.endswith('features.txt'):
                pet_files.append(os.path.join(atlas_path, file))
            elif file.endswith('time_series.csv'):
                ts_files.append(os.path.join(atlas_path, file))

        # Process Jij files
        jij_matrices = [np.loadtxt(f, delimiter=',') for f in jij_files]
        mean_jij_within_subject = np.mean(jij_matrices, axis=0) if jij_matrices else None

        # Process PET files
        pet_values = [np.loadtxt(f, delimiter=',') for f in pet_files]
        mean_pet_within_subject = np.mean(pet_values, axis=0) if pet_values else None

        # Process time series files and calculate empirical FC
        connectivity_measure = ConnectivityMeasure(kind='correlation')
        ts_matrices = [np.loadtxt(f, delimiter=',') for f in ts_files]
        empirical_fcs = [connectivity_measure.fit_transform([ts])[0] for ts in ts_matrices]
        mean_empirical_fc_within_subject = np.mean(empirical_fcs, axis=0) if empirical_fcs else None

        # Store means for the subject
        subjects_data[subject] = {
            'Jij': mean_jij_within_subject,
            'PET': mean_pet_within_subject,
            'Empirical_FC': mean_empirical_fc_within_subject
        }

    # Aggregate means across subjects and save
    for data_type in ['Jij', 'PET', 'Empirical_FC']:
        aggregated_data = np.array(
            [data[data_type] for _, data in subjects_data.items() if data[data_type] is not None])
        mean_aggregated_data = np.mean(aggregated_data, axis=0)

        if data_type != 'PET':  # Save as image and CSV for Jij and Empirical_FC
            save_matrix_as_image(mean_aggregated_data, f'Mean {data_type}',
                                 os.path.join(output_folder, f'mean_{data_type.lower()}.jpg'))
        else:  # For PET, plot values
            plt.figure()
            plt.plot(mean_aggregated_data)
            plt.title(f'Mean {data_type} Values')
            plt.savefig(os.path.join(output_folder, f'mean_{data_type.lower()}.jpg'))
            plt.close()

        save_matrix_as_csv(mean_aggregated_data, os.path.join(output_folder, f'mean_{data_type.lower()}.csv'))



# Example usage
base_folder = '/home/brainlab-qm/Desktop/New_test/To_Analyze_84'
output_folder = '/home/brainlab-qm/Desktop/Ising_test_10_03/To_Analyze_84/'
calculate_and_save_mean_matrices_within_subjects(base_folder, output_folder)

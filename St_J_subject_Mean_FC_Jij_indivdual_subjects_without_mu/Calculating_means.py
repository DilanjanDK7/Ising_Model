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

def calculate_and_save_mean_matrices(base_folder, output_folder):
    parcellations_data = {}

    subjects = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    for subject in subjects:
        subject_path = os.path.join(base_folder, subject)
        parcellations = [p for p in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, p))]

        for parcellation in parcellations:
            jij_files, pet_files, ts_files = [], [], []
            parcellation_path = os.path.join(subject_path, parcellation)

            for file in os.listdir(parcellation_path):
                if file.endswith('Jij.csv'):
                    jij_files.append(os.path.join(parcellation_path, file))
                elif file.endswith('features.txt'):
                    pet_files.append(os.path.join(parcellation_path, file))
                elif file.endswith('time_series.csv'):
                    ts_files.append(os.path.join(parcellation_path, file))

            # Process Jij files
            jij_matrices = [np.loadtxt(f, delimiter=',') for f in jij_files]
            mean_jij = np.mean(jij_matrices, axis=0) if jij_matrices else None

            # Process PET files
            pet_values = [np.loadtxt(f, delimiter=',') for f in pet_files]
            mean_pet = np.mean(pet_values, axis=0) if pet_values else None

            # Process time series files and calculate empirical FC
            connectivity_measure = ConnectivityMeasure(kind='correlation')
            ts_matrices = [np.loadtxt(f, delimiter=',') for f in ts_files]
            empirical_fcs = [connectivity_measure.fit_transform([ts])[0] for ts in ts_matrices]
            mean_empirical_fc = np.mean(empirical_fcs, axis=0) if empirical_fcs else None

            # Aggregate data for each parcellation
            if parcellation not in parcellations_data:
                parcellations_data[parcellation] = {'Jij': [], 'PET': [], 'Empirical_FC': []}

            if mean_jij is not None:
                parcellations_data[parcellation]['Jij'].append(mean_jij)
            if mean_pet is not None:
                parcellations_data[parcellation]['PET'].append(mean_pet)
            if mean_empirical_fc is not None:
                parcellations_data[parcellation]['Empirical_FC'].append(mean_empirical_fc)

    # Calculate mean across all subjects for each parcellation and save
    for parcellation, data in parcellations_data.items():
        parcellation_output_folder = os.path.join(output_folder, parcellation)
        if not os.path.exists(parcellation_output_folder):
            os.makedirs(parcellation_output_folder, exist_ok=True)

        # Jij
        if data['Jij']:
            mean_jij = np.mean(data['Jij'], axis=0)
            save_matrix_as_image(mean_jij, 'Mean Jij Matrix', os.path.join(parcellation_output_folder, 'mean_jij.jpg'))
            save_matrix_as_csv(mean_jij, os.path.join(parcellation_output_folder, 'mean_jij.csv'))

        # PET
        if data['PET']:
            mean_pet = np.mean(data['PET'], axis=0)
            plt.figure()
            plt.plot(mean_pet)
            plt.title('Mean PET Values')
            plt.savefig(os.path.join(parcellation_output_folder, 'mean_pet.jpg'))
            plt.close()
            save_matrix_as_csv(mean_pet, os.path.join(parcellation_output_folder, 'mean_pet.csv'))

        # Empirical FC
        if data['Empirical_FC']:
            mean_empirical_fc = np.mean(data['Empirical_FC'], axis=0)
            save_matrix_as_image(mean_empirical_fc, 'Mean Empirical FC', os.path.join(parcellation_output_folder, 'mean_empirical_fc.jpg'))
            save_matrix_as_csv(mean_empirical_fc, os.path.join(parcellation_output_folder, 'mean_empirical_fc.csv'))

# Example usage
base_folder = '/home/brainlab-qm/Desktop/Ising_test_10_03/Subject_files_parcellation'
output_folder = '/home/brainlab-qm/Desktop/Ising_test_10_03/Mean/sub-Sub1'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
calculate_and_save_mean_matrices(base_folder, output_folder)

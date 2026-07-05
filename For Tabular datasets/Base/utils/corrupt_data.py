import numpy as np
import pandas as pd
from config import num_clients, one_zero_dataset, continuous_datasets, percentage_of_corrupt, corrupted_clients
import os


def add_gaussian_noise(features_df, num_points_to_corrupt, std_dev=0.3):
    features = features_df.values.copy()  # Convert DataFrame to numpy array copy to make it writable
    noise = np.random.normal(0, std_dev, (num_points_to_corrupt, features.shape[1]))
    corrupt_indices = np.random.choice(features.shape[0], num_points_to_corrupt, replace=False)
    features[corrupt_indices] += noise
    features = np.clip(features, -1, 1)
    return pd.DataFrame(features, columns=features_df.columns, index=features_df.index)

def flip_labels(labels, num_points_to_flip):
    labels = np.array(labels, copy=True)  # Make a writable copy
    flip_indices = np.random.choice(len(labels), num_points_to_flip, replace=False)
    labels[flip_indices] = 1 - labels[flip_indices]  # Flip 0 to 1 and 1 to 0
    return labels

def introduce_outliers(features_df, num_samples_to_corrupt, fraction_of_features=0.3):
    features = features_df.values.copy()  # Convert DataFrame to numpy array copy to make it writable
    num_samples, num_features = features.shape
    sample_indices = np.random.choice(num_samples, num_samples_to_corrupt, replace=False)

    # Compute Q1, Q3, and IQR for each feature
    Q1 = np.percentile(features, 25, axis=0)
    Q3 = np.percentile(features, 75, axis=0)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    for row_idx in sample_indices:
        num_features_to_modify = int(fraction_of_features * num_features)
        feature_indices = np.random.choice(num_features, num_features_to_modify, replace=False)
        for col_idx in feature_indices:
            features[row_idx, col_idx] = upper_bound[col_idx] if np.random.rand() > 0.5 else lower_bound[col_idx]

    return pd.DataFrame(features, columns=features_df.columns, index=features_df.index)



def apply_noise_to_selected_clients(train_data_list, dataset):
    # Create a copy of train_data_list to avoid modifying the original
    modified_train_data_list = [df.copy() for df in train_data_list]
    
    # Select half of the clients to corrupt
    client_set = np.random.choice(num_clients, size=corrupted_clients, replace=False)
    
    # Initialize noise types
    noise_type_list = ['none'] * num_clients
    
    # Assign noise types for selected clients
    for i in client_set:
        if dataset in one_zero_dataset:
            noise_type_list[i] = 'flip_labels'
        elif dataset in continuous_datasets:
            noise_type_list[i] = np.random.choice(['flip_labels', 'gaussian', 'outliers'])
    
    # Apply noise to selected clients

    for i in client_set:
        df = modified_train_data_list[i]
        features = df.iloc[:, :-1].copy()  # All columns except last (labels)
        labels = df.iloc[:, -1].values.copy()  # Last column as numpy array
        
        if noise_type_list[i] == 'gaussian':
            num_points_to_corrupt = int(percentage_of_corrupt * features.shape[0])
            features = add_gaussian_noise(features, num_points_to_corrupt, std_dev=0.3)
        elif noise_type_list[i] == 'flip_labels':
            num_points_to_flip = int(percentage_of_corrupt * len(labels))
            labels = flip_labels(labels, num_points_to_flip)
        elif noise_type_list[i] == 'outliers':
            num_samples_to_corrupt = int(percentage_of_corrupt * features.shape[0])
            features = introduce_outliers(features, num_samples_to_corrupt)
        
        # Update the modified DataFrame
        modified_train_data_list[i] = pd.concat([features, pd.Series(labels, name=df.columns[-1], index=df.index)], axis=1)
    

    os.makedirs('results/noise_info', exist_ok=True)
    
    # Save noise_type_list to Excel
    noise_info_df = pd.DataFrame({
        'client_id': range(num_clients),
        'noise_type': noise_type_list
    })
    noise_info_df.to_excel(f'results/noise_info/{dataset}_noise_info.xlsx', index=False)

    return modified_train_data_list



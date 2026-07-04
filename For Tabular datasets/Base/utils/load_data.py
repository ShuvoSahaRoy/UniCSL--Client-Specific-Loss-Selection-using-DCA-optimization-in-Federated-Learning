import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .plot_result import plot_data_distribution
from config import continuous_datasets, SEED, non_iid, dataset_path, noise
from .utils import iid_distribution, noniid_distribution
from .corrupt_data import apply_noise_to_selected_clients



def prepare_client_data(data):
    if non_iid == 1:
        return noniid_distribution(data)
    else:
        return iid_distribution(data)


# Load Dataset Function for loading the dataset and partitioning it into training and testing data
def load_datasets(dataset):
    df = pd.read_csv(f"{dataset_path+dataset}.csv", header=None)
    print(f"Loaded {dataset} with shape: {df.shape}")

    if dataset in continuous_datasets:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    # Split into train-validation and global test data
    train_data, test_val_data = train_test_split(df, test_size=0.2,random_state=SEED, shuffle=True)
    print(f"Train-Val Data Shape: {train_data.shape} | Test Data Shape: {test_val_data.shape}")

    # Second split: train (60%) and val (20%) from the 80%
    test_data, val_data = train_test_split(test_val_data, test_size=0.5, random_state=SEED, shuffle=True)

    # check df.shape 

    train_data_list = prepare_client_data(train_data)
    # plot_data_distribution(train_data_list, title=f"Data Distribution for {dataset.split('.')[0]} Dataset")

    # clients_data = []
    # for client_data in train_data_list:
    #     client_train, client_val = train_test_split(client_data, test_size=0.1, random_state=SEED, shuffle=True)
    #     clients_data.append([client_train, client_val])

      # Default to 'none' to avoid index errors

    if noise:
        # this is modified train_data_list with noise applied to selected clients, original train_data_list is not modified
        train_data_list = apply_noise_to_selected_clients(train_data_list, dataset.split('.')[-1])

    global_test_data = [test_data, val_data]

    # check df.shape is equal to all distributed data
    # total_samples = sum(len(client[0]) + len(client[1]) for client in clients_data) + len(global_test_data[0]) + len(global_test_data[1])
    total_samples = sum(len(client) for client in train_data_list) + len(test_data) + len(val_data)
    assert total_samples == len(df), f"Sample mismatch: {total_samples} != {len(df)}"
    print(f"Total samples distributed: {total_samples} | Original dataset samples: {len(df)}")


    return train_data_list, global_test_data

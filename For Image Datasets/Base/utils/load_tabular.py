import pandas as pd
import numpy as np
import os, sys, time
import pickle
from sklearn.preprocessing import OrdinalEncoder
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

# Add project root to Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "../.."))
# sys.path.insert(0, project_root)

from Base.utils.corrupt_data import corrupt_tabular_data
from config import num_clients, non_iid, min_samples_per_client, SEED, val_fraction, corrupt_data, alpha, corrupted_clients, dataset_path


# --------------------------------------------------
# Main loader
# --------------------------------------------------
def load_tabulardataset(dataset_name):

    # --------------------------------------------------
    # Create save directory
    # --------------------------------------------------
    save_dir = "saved_datasets"
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # Determine partition name
    # --------------------------------------------------
    if non_iid == 0:
        partition_name = "iid"
    elif non_iid == 1 and not corrupt_data:
        partition_name = "noniid"
    elif non_iid == 1 and corrupt_data:
        partition_name = "noniid_corrupted"
    else:
        raise ValueError("Unsupported partition configuration")

    # --------------------------------------------------
    # Build versioned filename
    # --------------------------------------------------
    filename = (f"cifar10_{partition_name}_clients{num_clients}_seed{SEED}.pkl")
    save_path = os.path.join(save_dir, filename)

    # --------------------------------------------------
    # Load if exists
    # --------------------------------------------------
    if os.path.exists(save_path):
        print(f"Loading saved dataset: {filename}")
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["val"], data["test"]

    print(f"Saved file not found. Generating new distribution: {filename}")
    # --------------------------------------------------
    # Load dataset file
    # --------------------------------------------------
    # df = pd.read_csv(dataset_name)
    dataset_file = os.path.join(dataset_path, f"{dataset_name}.csv")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # ---------------- Partitioner ------------------
    if non_iid == 0:
        partitioner = IidPartitioner(num_partitions=num_clients)
    elif non_iid == 1:
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by="label",
            alpha=alpha,
            min_partition_size=min_samples_per_client,
            self_balancing=False,
        )
    else:
        raise ValueError("Only IID (0) and Dirichlet (1) supported")

    fds = FederatedDataset(
        dataset="csv",
        partitioners={"train": partitioner},
        data_files={"train": dataset_file},
        cache_dir="D:/datasets/huggingface/"
    )

    train_data_list = []
    val_data_list = []
    test_data_fragments = []  # Collect test fragments from each client

    client_noise_types = np.random.choice(["gaussian", "label_flip", "outliers"], size=corrupted_clients,replace=True)

    # ---------------- Client data ------------------
    for client_id in range(num_clients):
        # Load partition as pandas DataFrame
        s_time = time.time()
        df = fds.load_partition(client_id).with_format("pandas")[:]
        df.dropna(inplace=True)

        # Encode categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if "label" in categorical_cols:
            categorical_cols.remove("label")  # Don't encode the target
        
        if len(categorical_cols) > 0:
            enc = OrdinalEncoder()
            df[categorical_cols] = enc.fit_transform(df[categorical_cols])

        # Separate features and labels
        X = df.drop("label", axis=1)
        y = df["label"]

        # Rename label column to 'y' for consistency
        df_combined = X.copy()
        df_combined["y"] = y.values

        # Split into train+val (80%) and test (20%)
        test_df = df_combined.sample(frac=0.2).reset_index(drop=True)
        train_val_df = df_combined.drop(test_df.index).reset_index(drop=True)

        # Store test fragment for later merging
        test_data_fragments.append(test_df)

        if corrupt_data:
            train_val_df = corrupt_tabular_data(train_val_df,client_noise_types[client_id],corruption_rate=0.2)

        # Create validation split from the client's training data
        if val_fraction is not None and val_fraction > 0:
            val_df = train_val_df.sample(frac=val_fraction,).reset_index(drop=True)
            train_df = train_val_df.drop(val_df.index).reset_index(drop=True)
        else:
            val_df = train_val_df.iloc[0:0].copy()
            train_df = train_val_df

        train_data_list.append(train_df)
        val_data_list.append(val_df)

        print(f"{client_id} takes {time.time() - s_time}")
    # ---------------- Merge all test fragments into global test ------------------
    global_test_data = pd.concat(test_data_fragments, ignore_index=True)

    total_train = sum(len(client) for client in train_data_list)
    total_val = sum(len(v) for v in val_data_list)
    total_test = len(global_test_data)
    total_samples = total_train + total_val + total_test

    print(f"Total samples distributed: {total_samples} | Train: {total_train} | Val: {total_val} | Test: {total_test}")

    # --------------------------------------------------
    # Save everything in ONE versioned file
    # --------------------------------------------------
    with open(save_path, "wb") as f:
        pickle.dump({
            "train": train_data_list,
            "val": val_data_list,
            "test": global_test_data,
            "metadata": {
                "partition": partition_name,
                "alpha": alpha,
                "num_clients": num_clients,
                "val_fraction": val_fraction,
                "SEED": SEED,
                "dataset_name": dataset_name
            }
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved as: {filename}")

    return train_data_list, val_data_list, global_test_data


# --------------------------------------------------
# Run
# --------------------------------------------------
# num_clients = 100
# non_iid = 1
# min_samples_per_client = 20
# val_fraction = 0.1
# SEED = 123
# corrupt_data = 0
# corrupted_clients = num_clients
# alpha = 0.5
# dataset_name = "adult"
# train_data_list, val_data_list, global_test_data = load_tabulardataset()

# # Print totals
# total_train = sum(len(df) for df in train_data_list)
# total_val = sum(len(df) for df in val_data_list)
# total_test = len(global_test_data)
# print(f"\nTotal Train: {total_train}")
# print(f"Total Val: {total_val}")
# print(f"Total Test: {total_test}")
# print(f"Grand Total: {total_train + total_val + total_test}")
# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
import os
import pickle
 
from Base.utils.utils import corrupt_client_data, visualize_multiple_samples, visualize_corruption_examples 
from config import num_clients, non_iid, min_samples_per_client, SEED, val_fraction, corrupt_data, alpha, corrupted_clients


# --------------------------------------------------
# Transform: normalize + flatten
# --------------------------------------------------
pytorch_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def apply_transforms(batch):
    batch["image"] = [
        pytorch_transforms(img).view(-1).numpy()   # (784,)
        for img in batch["image"]
    ]
    return batch

# --------------------------------------------------
# HF Dataset → Pandas (flattened, scalar label)
# --------------------------------------------------
def hf_dataset_to_dataframe(dataset):
    X = []
    y = []

    for sample in dataset:
        X.append(sample["image"])     # (784,)
        y.append(sample["label"])     # scalar (0–9)

    X = np.vstack(X)                  # (n, 784)
    y = np.array(y)

    df_X = pd.DataFrame(
        X,
        columns=[f"x_{i}" for i in range(784)]
    )
    df_y = pd.DataFrame(y, columns=["y"])

    return pd.concat([df_X, df_y], axis=1)

# --------------------------------------------------
# Main loader
# --------------------------------------------------
def load_mnist():

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
    filename = (f"mnist_{partition_name}_clients{num_clients}_seed{SEED}.pkl")

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


    # ---------------- Partitioner ------------------
    if non_iid == 0:
        partitioner = IidPartitioner(num_partitions=num_clients)

    elif non_iid == 1:
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by="label",
            alpha=0.5,
            min_partition_size=min_samples_per_client,
            self_balancing=False,
        )
    else:
        raise ValueError("Only IID (0) and Dirichlet (1) supported")

    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
        cache_dir="D:/datasets/huggingface/"
    )

    train_data_list = []
    val_data_list = []

    client_noise_types = np.random.choice(["gaussian", "salt_pepper", "blur", "occlusion", "label_flip"],
                                          size=corrupted_clients,replace=True)
    # ---------------- Client data ------------------
    for client_id in range(num_clients):
        partition = fds.load_partition(client_id)
        partition = partition.with_transform(apply_transforms)

        train_df = hf_dataset_to_dataframe(partition)

        if corrupt_data:
            train_df = corrupt_client_data(train_df, client_noise_types[client_id], corruption_rate=0.2)
        # Create validation split from the client's training data
        if val_fraction is not None and val_fraction > 0:
            # Use a reproducible per-client seed so splits vary between clients
            val_df = train_df.sample(frac=val_fraction).reset_index(drop=True)
            train_df = train_df.drop(val_df.index).reset_index(drop=True)
        else:
            val_df = train_df.iloc[0:0].copy()

        train_data_list.append(train_df)
        val_data_list.append(val_df)

    # ---------------- Global test (no split) ------------------
    testset = fds.load_split("test").with_transform(apply_transforms)
    global_test_data = hf_dataset_to_dataframe(testset)

    total_train = sum(len(client) for client in train_data_list)
    total_val = sum(len(v) for v in val_data_list)
    total_samples = total_train + total_val + len(global_test_data)
    assert total_samples == 70000, f"Sample mismatch: {total_samples} != 70000"
    print(f"Total samples distributed: {total_samples} | Train: {total_train} | Val: {total_val} | Test: {len(global_test_data)}")


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
                "SEED": SEED
            }
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dataset saved as: {filename}")

    return train_data_list, val_data_list, global_test_data


# --------------------------------------------------
# Run
from matplotlib import pyplot as plt
def visualize_multiple_samples(train_data_list, client_id, n_samples=5):
    
    client_df = train_data_list[client_id]
    indices = np.random.choice(len(client_df), n_samples, replace=False)

    plt.figure(figsize=(10, 2))
    
    for i, idx in enumerate(indices):
        row = client_df.iloc[idx]
        image = row.drop("y").values.reshape(28, 28)
        label = int(row["y"])

        plt.subplot(1, n_samples, i+1)
        plt.imshow(image, cmap="gray")
        plt.title(label)
        plt.axis("off")

    plt.suptitle(f"Client {client_id}")
    plt.tight_layout()
    plt.show()
# --------------------------------------------------
# num_clients = 100
# non_iid = 1
# min_samples_per_client = 20
# val_fraction = 0.1
# SEED = 123
# train_data_list, val_data_list, global_test_data = load_mnist()
# # visualize_multiple_samples(train_data_list, client_id=10, n_samples=8)
# visualize_corruption_examples(train_data_list[55])
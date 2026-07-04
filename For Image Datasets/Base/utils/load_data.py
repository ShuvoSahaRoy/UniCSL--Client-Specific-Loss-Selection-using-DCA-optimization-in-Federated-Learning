from Base.utils.load_mnist import load_mnist
from Base.utils.load_fashionmnist import load_fashionmnist
from Base.utils.load_femnist import load_femnist
from Base.utils.load_tabular import load_tabulardataset
from config import *

def load_datasets(dataset_name):
    # print(f"SEED {SEED} non_iid {non_iid} noise {corrupt_data}\n")
    if dataset_name == "mnist":
        return load_mnist()
    elif dataset_name == "fashionmnist":
        return load_fashionmnist()
    elif dataset_name == "femnist":
        return load_femnist()
    elif dataset_name == "cifar10":
        return load_tabulardataset(dataset_name)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    



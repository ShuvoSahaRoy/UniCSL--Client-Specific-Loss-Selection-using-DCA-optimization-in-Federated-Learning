# UniCSL: Client-Specific Loss Selection using DCA Optimization in Federated Learning (Image Datasets)

This directory contains the implementation of **UniCSL** (Client-Specific Loss Selection using DCA optimization in Federated Learning) and various state-of-the-art federated learning baseline algorithms evaluated on image datasets (e.g., MNIST, FashionMNIST, CIFAR-10, FEMNIST).

---

## Table of Contents
1. [Prerequisites & Installation](#prerequisites--installation)
2. [Project Directory Structure](#project-directory-structure)
3. [Configuration (`config.py`)](#configuration-configpy)
4. [Running the Simulations](#running-the-simulations)
   - [Running `main.py` (Single Run/Algorithm)](#running-mainpy-single-runalgorithm)
   - [Running `run.py` (Batch Experiments)](#running-runpy-batch-experiments)
5. [Key Scripts Explained](#key-scripts-explained)
6. [Results & Logs](#results--logs)

---

## Prerequisites & Installation

To set up the project and install all required dependencies, ensure you have Python 3.11 installed. You can install all dependencies into your virtual environment using `pip`:

```bash
# 1. Activate your virtual environment (if applicable)
# Example (Windows):
# .\0venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Main Dependencies Included:
- **PyTorch & Torchvision**: Deep learning framework and datasets.
- **Flower (`flwr` & `flwr-datasets`)**: Federated learning dataset utilities.
- **Matplotlib & Seaborn**: For plotting and visualizing results.
- **OpenPyXL & Pandas**: For managing and saving simulation results to Excel spreadsheets.
- **OpenCV-Python (`opencv-python`)**: For image manipulation and corruption functions.
- **psutil**: For monitoring system resource/memory usage during simulation runs.

---

## Project Directory Structure

```
For Image Datasets/
│
├── config.py                  # Global simulation settings and hyperparameters
├── main.py                    # Main simulation entrypoint (runs single/all algos)
├── run.py                     # Automation script running experiments across seeds & data settings
├── requirements.txt           # Python dependency checklist
│
├── Base/                      # Core implementations
│   ├── UniCSL/                # Proposed Client-Specific Loss Selection (UniCSL)
│   │   ├── unicsl_static.py   # UniCSL static loss-selection simulation
│   │   └── utils/             # SVM helper solvers for UniCSL optimization
│   │
│   ├── baselines/             # SOTA Federated Learning Algorithms
│   │   ├── fedavg.py          # Federated Averaging (FedAvg)
│   │   ├── fedprox.py         # FedProx (handles system heterogeneity with proximal term)
│   │   ├── scaffold.py        # SCAFFOLD (addresses client drift with control variates)
│   │   ├── fednova.py         # FedNova (handles objective inconsistency due to local steps)
│   │   ├── feddyn.py          # FedDyn (dynamic regularization)
│   │   ├── fedopt.py          # FedOpt (server-side adaptive optimizers)
│   │   ├── pFedMe.py          # pFedMe (personalized FL using Moreau envelopes)
│   │   ├── loss_functions.py  # Definitions of candidate loss functions (Hinge, Smooth Hinge, Ramp, etc.)
│   │   └── ..._multi_loss.py  # Multi-loss variants for FedAvg and FedDyn
│   │
│   └── utils/                 # Dataset loading, corruption, and results plotting
│       ├── corrupt_data.py    # Implements label and feature noises
│       ├── load_data.py       # Helper interface for dataset selection
│       ├── load_mnist.py      # MNIST partitioning and dataset loaders
│       ├── load_femnist.py    # FEMNIST dataset loaders
│       ├── load_fashionmnist.py# Fashion-MNIST dataset loaders
│       └── utils.py           # Helper metrics, seeding, and logging functions
│
└── results/                   # Experimental output spreadsheets and figures
```

---

## Configuration (`config.py`)

`config.py` acts as the single source of truth for all simulation configurations and hyperparameters. 

### Key Config Variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `SEED` | `42` | Random seed to guarantee reproducibility across runs. |
| `num_clients` | `100` | Total number of clients participating in the network. |
| `participants` | `20` | Number of clients selected/sampled in each round. |
| `CR` | `50` | Number of communication rounds between the server and clients. |
| `local_epoch` | `10` | Number of local training epochs per client per round. |
| `batch_size` | `1000000000` | Batch size (set very high to run full-batch gradient descent). |
| `non_iid` | `0` | Toggle for data partitioning: `0` = IID (equal split); `1` = Non-IID (heterogeneous). |
| `alpha` | `0.5` | Dirichlet concentration parameter (lower value = higher non-IID skew). |
| `corrupt_data` | `0` | Data corruption toggle: `0` = clean data; `1` = corrupt data applied to clients. |
| `corrupted_clients`| `num_clients` | Total number of clients subject to data corruption. |
| `val_fraction` | `0.1` | Fraction of client training data used for local validation (critical for UniCSL). |
| `device` | Auto-detected | Selects GPU (`cuda`/`mps`) or `cpu` for execution. |
| `mu` | `0.1` | Regularization penalty strength for `FedProx` and `pFedMe`. |
| `alpha_coef` | `0.01` | Adaptive regularization penalty for `FedDyn`. |
| `all_datasets` | `['mnist']` | Target datasets for the run (supports `mnist`, `fashionmnist`, `cifar10`, `femnist`). |
| `aggregations` | `['fedavg', ...]` | Active algorithms to run during the simulation execution. |
| `save` | `1` | Save final accuracy metrics to Excel spreadsheet under `results/`. |
| `plot` | `0` | Plot and display convergence curve comparison graphs. |
| `save_log` | `0` | Toggle output redirection to log files. |

---

## Running the Simulations

### Running `main.py` (Single Run/Algorithm)

Use `main.py` directly when you want to run specific algorithms under a single set of environment parameters defined in `config.py`.

#### Run all active algorithms sequentially
This runs all the algorithms specified in the `aggregations` list in `config.py`:
```bash
python main.py
```


### Running `run.py` (Batch Experiments)

`run.py` is the main automation driver. It executes batch experiments across multiple seeds and heterogeneous settings automatically. 

#### When to run `run.py`:
Run `run.py` when you are ready to evaluate and compare the algorithms extensively (e.g., generating paper benchmarks).

#### What it does:
1. Loops over multiple random seeds: `SEEDS = [1234, 123, 12, 1, 42]`.
2. For each seed, it systematically generates 3 environmental configurations:
   * **IID & Clean**: `non_iid = 0, corrupt_data = 0`
   * **Non-IID & Clean**: `non_iid = 1, corrupt_data = 0`
   * **Non-IID & Corrupt**: `non_iid = 1, corrupt_data = 1`
3. Programmatically edits `config.py` using regex to switch these variables.
4. Spawns clean subprocesses running `main.py` for each algorithm under each configuration.
5. Saves execution times, validation results, and test accuracies into Excel spreadsheets under the `results/` folder.

#### Run command:
```bash
python run.py
```

---

## Key Scripts Explained

* **`main.py`**: Initializer script that orchestrates datasets loading, initializes client communication schedules, sets up the model (SVM structure by default), runs client-side training and server-side aggregation for each selected method, and saves output data.
* **`Base/UniCSL/unicsl_static.py`**: Implementation of the UniCSL static algorithm. It partitions validation data for every client to evaluate 10 candidate loss functions, runs DCA optimization to assign optimal client-specific loss functions, and trains global parameters.
* **`Base/baselines/loss_functions.py`**: Contains mathematical formulations and gradients for candidate loss functions:
  * Least Squares (LS)
  * Hinge / Squared Hinge / Smooth Hinge / Truncated Hinge
  * Ramp / Smoothed Ramp Loss
  * Non-convex Exponential Loss
* **`Base/utils/utils.py`**: General toolbox handling seed setup, Excel result aggregation, custom learning rate loaders, client schedules generator, and SVM weights logging.

---

## Results & Logs

Simulation outputs are systematically outputted to the `results/` folder:
1. **Excel Reports (`results/{SEED}_experimental_results.xlsx`)**: Stores testing accuracy histories for each round, allowing direct comparison of convergence speed and final accuracies.
2. **Frequency Counts (Printed to console/logs)**: UniCSL writes out tables summarizing how frequently each candidate loss function was selected by DCA optimization (e.g., least squares, smooth hinge, or non-convex exp loss), indicating optimal selections for clean versus noisy clients.

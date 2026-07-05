# UniCSL: Client-Specific Loss Selection using DCA Optimization in Federated Learning

This folder contains the federated learning simulation framework for tabular datasets. It compares various federated learning aggregation algorithms and UniCSL under different client distributions and noise corruption profiles.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Configuration (`config.py`)](#configuration-configpy)
4. [Running the Experiments](#running-the-experiments)
   - [Single Scenario: `main.py`](#1-single-scenario-mainpy)
   - [Benchmark Automation: `run.py`](#2-benchmark-automation-runpy)
5. [Results and Analysis](#results-and-analysis)

---

## Project Structure

```bash
├── Base/                     # Core helper modules
│   ├── classifier/           # PyTorch and NumPy classification models (SVM, LR)
│   ├── baselines/            # Baseline FL aggregators (FedAvg, FedProx, Scaffold, pFedMe, etc.)
│   ├── UniCSL/               # UniCSL optimization logic and variations
│   └── utils/                # Loaders, noise injection, and utility functions
├── datasets/                 # Tabular datasets (e.g. Run_or_walk_information, A9a, adult)
├── results/                  # Saved outputs, logs, and comparative charts
│   └── results_fed_analysis.ipynb # Jupyter notebook to compile results and build plots
├── config.py                 # Simulation hyperparameters and dataset selections
├── main.py                   # Entry point for running a single simulation run
├── run.py                    # Automator running experiments across multiple seeds
└── requirements.txt          # Python package dependencies
```

---

## Environment Setup

Ensure you have Python installed (3.11 to 3.13 are recommended). Follow these steps to set up your environment:

1. **Open your terminal** and navigate to the project: 
Open for tabular datasets folder then setup your envirnment

2. **Create a virtual environment**:
   ```powershell
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **PowerShell**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Command Prompt (cmd)**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - **Git Bash / Linux / macOS**:
     ```bash
     source .venv/Scripts/activate
     ```

4. **Install all required packages**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## Configuration (`config.py`)

All experimental parameters are defined in `config.py`. You can change these options directly before launching a run:

* **FL System Settings**:
  - `SEED`: Seed value (e.g., `1234`) to ensure reproducible dataset splits and noise assignments.
  - `num_clients`: The total number of simulated clients (default: `100`).
  - `participants`: Number of randomly sampled clients participating per round (default: `20`).
  - `CR`: Total number of communication rounds (default: `50`).
  - `local_epoch`: Local epochs run by clients before global aggregation (default: `10`).

* **Data Distribution Settings**:
  - `non_iid`: Set to `1` for Dirichlet-skewed non-IID class distribution, or `0` for IID distribution.
  - `alpha`: Dirichlet distribution parameter controlling non-IID skewness (default: `0.5`). Higher values represent more balanced distribution.
  - `min_samples_per_client`: Minimum number of samples allocated per client (default: `20`).

* **Noise & Corruption Settings**:
  - `noise`: Set to `1` to corrupt half of the client sets with different types of noise, or `0` for clean data.
  - `percentage_of_corrupt`: Fraction of corrupted samples inside a noisy client (default: `0.2`).
  - `corrupted_clients`: Number of clients to inject noise into (default: `num_clients`).

* **Aggregator-Specific Settings**:
  - `mu`: Proximal term coefficient for `FedProx` and `pFedMe` (default: `0.1`).
  - `alpha_coef`: Regularization coefficient for `FedDyn` (default: `0.1`).

* **Execution Filters**:
  - `all_datasets`: List of tabular datasets to run (e.g. `['Run_or_walk_information']`).
  - `aggregations`: List of FL algorithms/loss configurations to evaluate in the loop (e.g. `fedavg_sq_hinge`, `fedprox_sq_hinge`, `scaffold_sq_hinge`, `feddyn_sq_hinge`, `fedopt_sq_hinge`, `pfedme_sq_hinge`, `unicsl_static`).

---

## Running the Experiments

Depending on your objective, you will run either `main.py` or `run.py`.

### 1. Single Scenario: `main.py`

Run `main.py` when you want to execute the simulation for the current settings defined in `config.py` (e.g. a specific seed, a single dataset, and specific algorithms).

* **When to use**: Quick debugging, checking if code works, tuning parameters on a specific setup, or evaluating a single algorithm variation.
* **Execution**:
  ```powershell
  python main.py
  ```

---

### 2. Benchmark Automation: `run.py`

Run `run.py` when you want to perform a full, rigorous benchmark evaluation. 

* **What it does**: It loops through multiple seeds (`[1234, 123, 12, 1, 42]`) and automatically updates `config.py` to evaluate each seed under three scenarios:
  1. **IID, Clean**: `non_iid = 0`, `noise = 0`
  2. **Non-IID, Clean**: `non_iid = 1`, `noise = 0`
  3. **Non-IID, Corrupted**: `non_iid = 1`, `noise = 1`
* **When to use**: Generating final comparative experimental tables and data for papers/reports.
* **Execution**:
  ```powershell
  python run.py
  ```

---

## Results and Analysis

All outputs generated by the simulations are organized in the `results` folder:

1. **Experimental Data Tables**:
   Running the simulation generates Excel files in the format `results/{SEED}_experimental_results.xlsx`. This file stores accuracy metrics, execution time, and memory usage for each algorithm on the selected datasets.

2. **Noise and Corruptions Logging**:
   When client noise is enabled, details of which client received which noise pattern (e.g., `gaussian`, `flip_labels`, `outliers`) are saved to `results/noise_info/{dataset}_noise_info.xlsx`.

3. **Final Comparison and Visualization**:
   The Jupyter Notebook **`results/results_fed_analysis.ipynb`** aggregates and parses all Excel sheets produced across the seeds and scenarios. Running this notebook compiles the raw experimental metrics, executes statistical evaluations, and generates the final comparative graphs/tables showing performance rankings of the algorithms.

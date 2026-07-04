from config import save, non_iid, plot

import numpy as np, os
import pandas as pd

if save:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_data_distribution(client_datasets,title):
    client_names = [f"Client {i+1}" for i in range(len(client_datasets))]

    class_labels = np.unique(np.concatenate([df.iloc[:, -1].values for df in client_datasets]))
    counts_per_client = []

    for df in client_datasets:
        class_counts = df.iloc[:, -1].value_counts()
        counts_per_client.append(class_counts)

    fig, ax = plt.subplots(figsize=(12, 8))
    left = np.zeros(len(client_datasets))
    colors = plt.cm.Paired(np.linspace(0, 1, len(class_labels)))

    for i, class_label in enumerate(class_labels):
        class_counts = [counts.get(class_label, 0) for counts in counts_per_client]
        ax.barh(client_names, class_counts, left=left, label=f"Class {class_label}", color=colors[i])
        left += class_counts

    ax.set_ylabel("Clients")
    ax.set_xlabel("Number of Samples")
    ax.set_title("Data Distribution Across Clients and Classes")
    ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.title(title)
    if save:
        save_dir = 'results/data_distribution'
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"Saved plot to: {filename}")

    plt.show()



def plot_graph(global_accuracies, federated_accuracies, dataset, plot):
    if plot and len(federated_accuracies) == 0:
        rounds = list(range(len(global_accuracies)))  # Start from 0
        plt.close('all')
        plt.figure(figsize=(10, 6))

        # Plot Global Accuracy
        plt.plot(rounds, global_accuracies, linestyle='-', marker='o', color='b', label='Global Accuracy')

        # Set y-axis limits dynamically
        accuracy_min = min(global_accuracies)
        accuracy_max = max(global_accuracies) + 5  # Buffer for readability

        y_min = max(0, int((accuracy_min // 10) * 10)- 20)  # Ensure integer
        y_max = min(100, int(((accuracy_max + 9) // 10) * 10)+10)  # Ensure integer

        plt.ylim(y_min, y_max)
        plt.yticks(range(int(y_min), int(y_max) + 1, 10))  # Convert to int for range()

        # X-axis: Extend slightly beyond the last round for visibility
        plt.xlim(0, len(rounds) - 0.5)  # Slightly extend past last point

        # Labels and Title
        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Global Accuracy Over Rounds for {dataset}" if dataset else "Global Accuracy Over Rounds")

        # Improved legend positioning
        plt.legend(loc='lower right')

        # Add grid for better visualization
        plt.grid(True, linestyle=(0, (1, 1)), color='g', alpha=0.6)

        # Adjust layout to prevent cutoff
        plt.tight_layout()

        # Show the plot
        plt.show(block=True)
    
    elif plot and len(federated_accuracies)!=0:  # Use 'is not None' for clarity
        # Ensure both lists have the same length
        min_length = min(len(federated_accuracies), len(global_accuracies))
        rounds = list(range(min_length))  # Start from 0, use min length to avoid mismatch
        plt.close('all')
        plt.figure(figsize=(10, 6))
        
        # Plot Federated Accuracy
        plt.plot(rounds, federated_accuracies[:min_length], linestyle='-', color='g', label='Federated Accuracy')
        
        # Plot Global Accuracy
        plt.plot(rounds, global_accuracies[:min_length], linestyle='--', color='b', label='Global Accuracy')

        # Dynamically set y-axis range based on accuracies
        y_min = min(min(federated_accuracies[:min_length]), min(global_accuracies[:min_length]))
        y_max = max(max(federated_accuracies[:min_length]), max(global_accuracies[:min_length]))

        # Ensure y_min is >= 0 and rounded down to the nearest 10
        y_min = max(0, int((y_min // 10) * 10 - 20))  # Ensure integer

        # Ensure y_max is <= 100 and rounded **UP** to the nearest 20
        y_max = min(100, int(((y_max + 19) // 20) * 20))  # Ensure integer

        plt.ylim(y_min, y_max)
        plt.yticks(range(int(y_min), int(y_max) + 1, 10))  # Convert to int for range()

        # X-axis: Extend slightly beyond the last round for visibility
        plt.xlim(0, len(rounds) - 0.5)  # Slightly extend past last point

        # Axis labels and title
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy Over Communication Rounds for {dataset}' if dataset else 'Accuracy Over Communication Rounds')

        # Add legend to differentiate the lines
        plt.legend(loc='lower right')

        # Grid for better readability
        plt.grid(True)

        # Adjust layout to prevent cutoff
        plt.tight_layout()

        # Show the plot
        plt.show(block=True)

    else:
        pass


DATASET_NAME_MAP = {
    "2dplanes": "2D Planes",
    "a9a": "A9a",
    "adult": "Adult",
    "covtype": "Covtype",
    "phishing": "Phishing",
    "magic_gamma_telescope": "Magic Gamma Telescope",
    "fried": "Fried",
    "run_or_walk_information": "Run or Walk Information",
    "skin_nonskin": "Skin Non-skin",
    "hepmass": "Hepmass",
    "susy": "Susy",
    "qsar_oral_toxicity": "QSAR Oral Toxicity",

}
def format_dataset_name(dataset_name):
    return DATASET_NAME_MAP.get(dataset_name, dataset_name) 


# y_min = max(0, int((min(min(alg) for alg in accuracy_lists) // 10) * 10 - 10))
# y_max = min(100, int(((max(max(alg) for alg in accuracy_lists) + 9) // 10) * 10))


def plot_all(accuracy_lists, algo_names, dataset):
    """
    Plots accuracy trends for multiple algorithms with clean, minimal visuals.

    Parameters:
    - accuracy_lists: List of accuracy lists (one per algorithm).
    - algo_names: List of corresponding algorithm names.
    - dataset: Name of the dataset.
    - lr: Learning rate.
    - y_label: Label for y-axis (default: Accuracy).
    """
    plt.close('all')
    plt.figure(figsize=(8, 6))
    dataset = format_dataset_name(dataset)  

    colors = plt.cm.tab10.colors
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', '*', 'P']

    max_rounds = max(len(alg) for alg in accuracy_lists)

    # X-tick setup: multiples of 10, and ensure last point is included
    step =5 # max(5, (max_rounds // 5))
    x_ticks = list(range(0, max_rounds, step))
    if (max_rounds - 1) not in x_ticks:
        x_ticks.append(max_rounds - 1)


    for idx, (algo_data, algo_name) in enumerate(zip(accuracy_lists, algo_names)):
        x_values = list(range(len(algo_data)))
        marker_every = max(len(algo_data) // 5, 1)

        plt.plot(
            x_values,
            algo_data,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            markevery=marker_every,
            markersize=9,
            linewidth=2,
            label=algo_name
        )

    plt.xlabel("Communication Round", fontsize=24)
    plt.ylabel(f"Test Accuracy (%)", fontsize=24)
    # plt.title(f"Performance on {dataset}")

    plt.xticks(x_ticks)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(loc='lower right', fontsize=22)
    plt.grid(False)
    plt.tight_layout()
    # Save the plot if required
    if save:
        filename = f"results/{dataset}_{non_iid}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    if plot:
        plt.show(block=True)


def plot_lr_tuning(accuracies, lr_values, dataset, aggregation):
    """
    Plot accuracy trajectories for different learning rates.
    
    Args:
        accuracies (list): List of accuracy lists, one per lr value.
        lr_values (list): List of learning rate values corresponding to accuracies.
        dataset (str): Name of the dataset for the title.
        aggregation (str): Algorithm name for the title (e.g., 'FedAvg', 'UniCSL').
    """
    plt.close('all')
    plt.figure(figsize=(12, 6))  # Slightly wider for better visibility

    # Plot each accuracy list with a unique label for lr
    for i, acc_list in enumerate(accuracies):
        x = list(range(len(acc_list)))  # X-axis for this specific accuracy list
        plt.plot(x, acc_list, linestyle='-', label=f'lr={lr_values[i]}', alpha=0.7)

    # Calculate y-axis limits across all accuracy lists
    y_min = min(min(acc) for acc in accuracies if acc)  # Handle empty lists
    y_max = max(max(acc) for acc in accuracies if acc) + 5  # Buffer for readability

    # Ensure y_min is >= 0 and rounded down to the nearest 10
    y_min = max(0, int((y_min // 10) * 10))
    # Ensure y_max is <= 100 and rounded up to the nearest 10
    y_max = min(100, int(((y_max + 9) // 10) * 10))

    plt.ylim(y_min, y_max)

    # X-axis: Span the maximum length across all lists
    max_rounds = max(len(acc) for acc in accuracies if acc)  # Handle empty lists
    plt.xlim(0, max_rounds - 0.5)  # Slightly extend for visibility

    # Labels and title
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{aggregation} Accuracy vs. Communication Rounds for {dataset} (LR Tuning)")

    # Legend
    plt.legend(loc='lower right', title="Learning Rates")

    # Grid for better visibility
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(f"results/parameter_tunning_pfedme/{aggregation}_{format_dataset_name(dataset)}_{non_iid}.png", dpi=200, bbox_inches='tight')
    # Show the plot
    # plt.show()

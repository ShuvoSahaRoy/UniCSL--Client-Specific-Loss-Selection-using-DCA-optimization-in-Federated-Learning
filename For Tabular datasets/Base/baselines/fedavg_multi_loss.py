"""
FedAvg with Multi-Loss Selection for Logistic Regression.

Each client evaluates ALL available loss functions on their local data
using the validation set's F1 score.  The best-performing loss is locked
for that client for all subsequent communication rounds (static selection,
same strategy as unicsl_static but applied to logistic regression + SGD).

Loss functions are imported from loss_functions.py in the same folder.
"""

import numpy as np
from config import num_clients, CR, local_epoch
from Base.baselines.loss_functions import LOSS_FUNCTIONS, sigmoid
from Base.utils.utils import SERVER
from sklearn.metrics import f1_score
import tracemalloc
import time


# ──────────────────────────────────────────────────────────────────────
# Local training with a pluggable loss function
# ──────────────────────────────────────────────────────────────────────

def train_with_loss(model, data, lr, loss_fn):
    """Train logistic regression on *data* using *loss_fn* for local_epoch steps.

    Parameters
    ----------
    model    : np.ndarray   – [w1, w2, ..., wn, b]  (weights + bias)
    data     : DataFrame    – last column is the label
    lr       : float        – learning rate
    loss_fn  : object       – must expose .gradient(z, y) -> dL/dz

    Returns
    -------
    np.ndarray – updated model parameters [w | b]
    """
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    w = model[:-1].copy()
    b = model[-1]
    n = X.shape[0]

    for _ in range(local_epoch):
        z = np.dot(X, w) + b
        dL_dz = loss_fn.gradient(z, y)            # shape (n,)

        dw = (1.0 / n) * np.dot(X.T, dL_dz)
        db = (1.0 / n) * np.sum(dL_dz)

        w = w - lr * dw
        b = b - lr * db

    return np.concatenate((w, [b]))


# ──────────────────────────────────────────────────────────────────────
# Testing / F1 evaluation
# ──────────────────────────────────────────────────────────────────────

def test_model(model, test_data):
    """Return (accuracy%, f1_score) for the given model on test_data."""
    w = model[:-1]
    b = model[-1]
    X = test_data.iloc[:, :-1].values
    y = test_data.iloc[:, -1].values

    z = np.dot(X, w) + b
    y_pred = (sigmoid(z) > 0.5).astype(int)

    accuracy = np.mean(y_pred == y) * 100.0
    f1 = f1_score(y, y_pred, pos_label=1, average='binary', zero_division=0)
    return accuracy, f1


# ──────────────────────────────────────────────────────────────────────
# Per-client update  (with loss selection on first participation)
# ──────────────────────────────────────────────────────────────────────

def client_update(global_model, client_data, val_data, lr, client_loss_info):
    """
    If the client has not yet chosen a loss function (best_loss is None),
    try every registered loss, pick the one with the highest F1, and lock it.
    Otherwise, train using the already-selected loss.

    Parameters
    ----------
    global_model     : np.ndarray   – current global model
    client_data      : DataFrame    – this client's local training data
    val_data         : DataFrame    – validation data for F1 evaluation
    lr               : float        – learning rate
    client_loss_info : dict         – {'best_loss': None | str, 'best_instance': None | obj}

    Returns
    -------
    (updated_model, n_samples), selected_loss_name
    """

    if client_loss_info['best_loss'] is None:
        # ---- First time: evaluate every loss function ----
        best_f1 = -1.0
        best_model = None
        best_name = None
        best_instance = None

        for entry in LOSS_FUNCTIONS:
            loss_fn = entry['instance']
            candidate = train_with_loss(global_model.copy(), client_data, lr, loss_fn)
            _, f1 = test_model(candidate, val_data)

            if f1 > best_f1:
                best_f1 = f1
                best_model = candidate
                best_name = entry['name']
                best_instance = loss_fn

        return (best_model, client_data.shape[0]), best_name, best_instance

    else:
        # ---- Subsequent rounds: use locked loss ----
        loss_fn = client_loss_info['best_instance']
        updated = train_with_loss(global_model.copy(), client_data, lr, loss_fn)
        return (updated, client_data.shape[0]), client_loss_info['best_loss'], loss_fn


# ──────────────────────────────────────────────────────────────────────
# Main federated loop
# ──────────────────────────────────────────────────────────────────────

def main_fedavg_multi_loss(global_parameters, client_schedule,
                           train_data_list, test_data, val_data, lr):
    """
    FedAvg with per-client multi-loss selection.

    Parameters
    ----------
    global_parameters : np.ndarray    – initial model  [w | b]
    client_schedule   : list[array]   – selected client ids per round
    train_data_list   : list[DataFrame] – one DataFrame per client
    test_data         : DataFrame     – global test set (for accuracy tracking)
    val_data          : DataFrame     – validation set  (for F1-based loss selection)
    lr                : float         – learning rate

    Returns
    -------
    (global_accuracy_list, total_time, peak_memory_MB), loss_dictionary
    """

    # Book-keeping: which loss each client selected
    loss_dictionary = {
        cid: {'best_loss': None, 'best_instance': None}
        for cid in range(num_clients)
    }

    global_accuracy = []

    start_time = time.time()
    tracemalloc.start()

    for rnd in range(CR):
        client_set = client_schedule[rnd]

        n_total = 0
        local_updates = []

        for client in client_set:
            client_data = train_data_list[client]

            (updated_model, n_samples), loss_name, loss_instance = client_update(
                global_parameters.copy(), client_data, val_data, lr,
                loss_dictionary[client]
            )

            # Lock the chosen loss for this client
            loss_dictionary[client]['best_loss'] = loss_name
            loss_dictionary[client]['best_instance'] = loss_instance

            local_updates.append((updated_model, n_samples))
            n_total += n_samples

        # ---- FedAvg aggregation ----
        global_parameters = SERVER(global_parameters, local_updates, n_total)

        # ---- Evaluate global model ----
        acc, _ = test_model(global_parameters, test_data)
        global_accuracy.append(acc)
        print(f"[FedAvg-MultiLoss] Round {rnd + 1}/{CR} | Acc: {acc:.2f}")

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - start_time

    return (global_accuracy, total_time, peak_memory / (1024 ** 2)), loss_dictionary

import numpy as np
from config import *
from Base.utils.utils import SERVER
import tracemalloc
import time


# ──────────────────────────────────────────────────────────────────────
# Squared Hinge Loss — train & test (labels in {-1, +1})
# L     = max(0, 1 - y * z)^2
# dL/dz = -2 * y * max(0, 1 - y * z)
# ──────────────────────────────────────────────────────────────────────

def _convert_labels(y):
    """Convert labels from {0, 1} to {-1, +1}."""
    return 2.0 * y - 1.0


def _sq_hinge_test(model_parameters, test_dataset):
    """Test using sign(z) decision rule (labels in {-1, +1})."""
    w = model_parameters[:-1]
    b = model_parameters[-1]
    X = test_dataset.iloc[:, :-1].values
    y = _convert_labels(test_dataset.iloc[:, -1].values)  # {0,1} -> {-1,+1}

    z = np.dot(X, w) + b
    predictions = np.sign(z)

    # Treat z == 0 as +1 (convention)
    predictions[predictions == 0] = 1.0

    correct = np.sum(predictions == y)
    return (correct / len(y)) * 100


# ──────────────────────────────────────────────────────────────────────
# FedProx with Squared Hinge Loss
# ──────────────────────────────────────────────────────────────────────

def client_update(model, data, lr):
    global_model = model.copy()

    X = data.iloc[:, :-1].values
    y = _convert_labels(data.iloc[:, -1].values)   # {0,1} -> {-1,+1}
    w = model[:-1]
    b = model[-1]

    n = X.shape[0]

    for _ in range(local_epoch):
        z = np.dot(X, w) + b
        violation = np.maximum(0.0, 1.0 - y * z)   # max(0, 1 - y*z)
        grad_z = -2.0 * y * violation               # dL/dz

        dw = (1.0 / n) * np.dot(X.T, grad_z)
        db = (1.0 / n) * np.sum(grad_z)

        # ✅ L2 regularization
        dw += 2.0 * 1e-4 * w

        # Proximal term added to the gradients (FedProx modification)
        dw += mu * (w - global_model[:-1])
        db += mu * (b - global_model[-1])

        w = w - lr * dw
        b = b - lr * db

    model = np.concatenate((w, [b]))

    # del_w = global_model - model
    del_w = model
    local_update = (del_w, n)
    return local_update


def main_fedprox_sq_hinge(global_parameters, client_schedule, train_data_list, test_data, lr):

    global_accuracy = []
    start_time = time.time()
    tracemalloc.start()

    for round in range(CR):

        client_set = client_schedule[round]
        n = 0
        local_updates = []

        for client in client_set:
            client_data = train_data_list[client]

            local_update = client_update(global_parameters.copy(), client_data, lr)

            local_updates.append(local_update)
            n = n + client_data.shape[0]

        global_parameters = SERVER(global_parameters, local_updates, n)

        global_acc = _sq_hinge_test(global_parameters, test_data)
        global_accuracy.append(global_acc)
        print(f"[FedProx-SqHinge] Round {round + 1}/{CR} | Acc: {global_acc:.2f}")

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time

    return (global_accuracy, total_time, (peak_memory / (1024**2)))

import tracemalloc
import numpy as np
from config import num_clients, participants, CR, local_epoch
from Base.utils.utils import SERVER
import time, tracemalloc


# ──────────────────────────────────────────────────────────────────────
# Squared Hinge Loss — helpers (labels in {-1, +1})
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
    y = _convert_labels(test_dataset.iloc[:, -1].values)

    z = np.dot(X, w) + b
    predictions = np.sign(z)
    predictions[predictions == 0] = 1.0

    correct = np.sum(predictions == y)
    return (correct / len(y)) * 100


# ──────────────────────────────────────────────────────────────────────
# SCAFFOLD with Squared Hinge Loss
# ──────────────────────────────────────────────────────────────────────

def client_update(model, data, control_variate, global_control_variates, lr):
    """
    return : del_w, del_c, number_of_sample, updated_client_control_variates
    """
    global_model = model.copy()

    number_of_sample = data.shape[0]
    X = data.iloc[:, :-1].values
    y = _convert_labels(data.iloc[:, -1].values)   # {0,1} -> {-1,+1}

    for i in range(local_epoch):
        w = model[:-1]
        b = model[-1]
        z = np.dot(X, w) + b
        violation = np.maximum(0.0, 1.0 - y * z)
        grad_z = -2.0 * y * violation

        dw = (1.0 / number_of_sample) * np.dot(X.T, grad_z)
        db = (1.0 / number_of_sample) * np.sum(grad_z)

        # ✅ L2 regularization
        dw += 2.0 * 1e-4 * w

        gradient = np.concatenate((dw, [db]))

        corrected_gradient = gradient - control_variate + global_control_variates
        model = model - lr * corrected_gradient

    del_w = global_model - model

    # Update the local control variate
    new_control_variate = control_variate - global_control_variates + (global_model - model) * (1 / (lr * local_epoch))
    del_c = new_control_variate - control_variate

    return del_w, del_c, new_control_variate

# global_parameters, client_updates, lr, n, client_control_updates, global_control_variates
def SERVER_SCAFFOLD(global_model, local_updates, client_control_updates, global_control_variates, server_lr=1.0):

    delta_x = np.mean([w for w, _ in local_updates], axis=0)
    global_model = global_model - server_lr * delta_x

    mean_delta_c = np.mean(client_control_updates, axis=0)
    # Update the global control variate
    global_control_variates = global_control_variates + mean_delta_c * (participants / num_clients)

    return global_model, global_control_variates


def main_scaffold_sq_hinge(global_parameters, client_schedule, train_data_list, test_data, lr):

    features = train_data_list[0].shape[1]
    client_control_variates = np.zeros((num_clients, features))
    global_control_variates = np.zeros_like(global_parameters)

    global_accuracy = []

    start_time = time.time()
    tracemalloc.start()
    for round in range(CR):

        client_set = client_schedule[round]
        n = 0
        client_updates = []
        client_control_updates = []

        for client in client_set:
            client_data = train_data_list[client]

            del_w, del_c, new_c_client = client_update(global_parameters.copy(), client_data, client_control_variates[client], global_control_variates, lr)

            client_control_variates[client] = new_c_client

            client_updates.append((del_w, client_data.shape[0]))
            client_control_updates.append(del_c)
            n = n + client_data.shape[0]


        global_parameters, global_control_variates = SERVER_SCAFFOLD(global_parameters, client_updates,
                                                            client_control_updates, global_control_variates)

        global_acc = _sq_hinge_test(global_parameters, test_data)
        global_accuracy.append(global_acc)
        print(f"[SCAFFOLD-SqHinge] Round {round + 1}/{CR} | Acc: {global_acc:.2f}")

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time

    return (global_accuracy, total_time, (peak_memory / (1024**2)))

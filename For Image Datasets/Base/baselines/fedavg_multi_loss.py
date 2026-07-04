import numpy as np
from config import local_epoch, device, batch_size, CR, num_clients
import torch, copy
from torch.utils.data import DataLoader, TensorDataset
import time
import psutil
import os
import gc
from .fedavg import SERVER
from .loss_functions import get_loss_fn
from sklearn.metrics import f1_score

loss_functions_list = [
    {'Least Squares': [2]},
    {'Smooth Hinge': [10]},
    {'Squared Hinge': [0]},
    {'Truncated SH': [2]},
    {'Truncated LS': [2]},
    {'Smoothed Ramp1': [2]},
    {'Smoothed Ramp2': [2, 10]},
    {'nonconvex exp loss1': [2, 2, 3]},
    {'nonconvex exp loss2': [2, 2, 4]},
    {'nonconvex exp loss3': [2, 3, 4]}
]


@torch.no_grad()
def test_ovo(model, test_data):
    model.eval()

    # --- Prepare dataloader ---
    if isinstance(test_data, DataLoader):
        loader = test_data
    else:
        X = torch.tensor(test_data.drop(columns=["y"]).values, dtype=torch.float32)
        y = torch.tensor(test_data["y"].values, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)  # (N, K)

        votes = torch.zeros(x_batch.size(0), model.num_classes, device=device)

        # --- OvO voting ---
        for k, (i, j) in enumerate(zip(model.pair_i.tolist(),
                                       model.pair_j.tolist())):
            pred = logits[:, k]
            votes[pred > 0, i] += 1
            votes[pred <= 0, j] += 1

        preds = votes.argmax(dim=1)

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        # store for F1
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

    # --- Accuracy ---
    accuracy = (correct / total) * 100

    # --- F1 Score ---
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    f1 = f1_score(all_labels, all_preds, average='macro')  
    # use 'macro', 'weighted', or 'micro' depending on your need

    return accuracy, f1


def train_fedavg(model, data, lr, loss_name, loss_params):
    """
    Vanilla FedAvg client update using a specific loss function
    """
    device = next(model.parameters()).device
    model = copy.deepcopy(model)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = get_loss_fn(loss_name)

    # --- Make data minibatch-compatible ---
    # If data is already a DataLoader → use it
    if isinstance(data, DataLoader):
        loader = data
        n_samples = len(data.dataset)
    else:
        # Pandas DataFrame → tensors
        X = torch.tensor(data.drop(columns=["y"]).values, dtype=torch.float32)
        y = torch.tensor(data["y"].values, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        n_samples = len(dataset)

    # --- Local training ---
    for _ in range(local_epoch):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)              # (N, K)
            
            # Certain loss functions do not accept additional arguments
            if loss_name in ['Least Squares', 'Squared Hinge']:
                loss = loss_fn(logits, y_batch.long(), model.pair_i, model.pair_j)
            else:
                loss = loss_fn(logits, y_batch.long(), model.pair_i, model.pair_j, *loss_params)

            loss.backward()
            optimizer.step()

    return model, n_samples


def client_update(global_model, data, lr, loss_dict, val_data):
    # Check all loss functions initially for this client
    if loss_dict['best_loss'] is None:
        # best_accuracy = -1
        best_f1 = -1
        best_model = None
        best_loss = None
        best_n_samples = 0
        
        for loss_cfg in loss_functions_list:
            loss_name = list(loss_cfg.keys())[0]
            loss_params = loss_cfg[loss_name]
            
            # Train model locally with this loss
            temp_model, n_samples = train_fedavg(global_model, data, lr, loss_name, loss_params)
            
            # Evaluate on validation data
            accuracy, f1 = test_ovo(temp_model, val_data)
            
            if f1 > best_f1:
                # best_accuracy = accuracy
                best_f1 = f1
                best_model = temp_model
                best_loss = loss_cfg
                best_n_samples = n_samples
                
        local_model = best_model
        return local_model, best_n_samples, best_loss
    else:
        # Best loss is already assigned for this client
        best_loss = loss_dict['best_loss']
        loss_name = list(best_loss.keys())[0]
        loss_params = best_loss[loss_name]
        
        local_model, n_samples = train_fedavg(global_model, data, lr, loss_name, loss_params)
        return local_model, n_samples, best_loss


def main_fedavg_multi_loss(global_parameters, train_data_list, val_data_list, test_data, client_schedule, lr):

    global_accuracy = []
    
    # Track the best loss function dict for each client
    loss_dictionary = {client_id: {"best_loss": None} for client_id in range(num_clients)}

    # ---------------- PROFILING START ----------------
    process = psutil.Process(os.getpid())
    gc.collect()

    mem_start = process.memory_info().rss / (1024**2)  # MB
    time_start = time.perf_counter()
    # --------------------------------------------------

    for round in range(CR):

        client_set = client_schedule[round]
        local_models = []
        n = 0

        for client in client_set:
            client_data = train_data_list[client]
            val_data = val_data_list[client]

            local_model, n_i, best_loss = client_update(global_parameters, client_data, lr, loss_dictionary[client], val_data)

            # Assign the found best loss function to be preserved for subsequent rounds
            loss_dictionary[client]['best_loss'] = best_loss
            local_models.append((local_model, n_i))
            n += n_i

        global_parameters = SERVER(global_parameters, local_models, n)
        global_acc, _ = test_ovo(global_parameters, test_data)
        global_accuracy.append(global_acc)

        print(f"Round {round} FedAvg Multi-Loss Acc {global_acc:.2f}")

    # ---------------- PROFILING END ----------------
    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start
    # ------------------------------------------------
    print(f"FedAvg Multi-Loss Total Time: {total_time:.2f} seconds")
    print(f"FedAvg Multi-Loss Memory Used: {memory_used:.2f} MB")

    return (global_accuracy , total_time, memory_used), loss_dictionary

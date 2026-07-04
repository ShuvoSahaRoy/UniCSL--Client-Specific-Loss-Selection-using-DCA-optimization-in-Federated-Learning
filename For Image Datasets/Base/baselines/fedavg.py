import numpy as np
from config import local_epoch, device, batch_size, CR
import torch, copy
from torch.utils.data import DataLoader, TensorDataset
import time
import psutil
import os
import gc

@torch.no_grad()
def test_ovo(model, test_data):
    """
    Original OvO testing with explicit pairwise voting

    Args:
        model: LinearSVM_OVO
        test_data: (X_test, y_test) or DataLoader
        device: torch device (optional)
    """
    model.eval()

    # --- Prepare dataloader ---
    if isinstance(test_data, DataLoader):
        loader = test_data
    else:
        # Pandas DataFrame → tensors
        X = torch.tensor(test_data.drop(columns=["y"]).values, dtype=torch.float32)
        y = torch.tensor(test_data["y"].values, dtype=torch.long)

        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)  # (N, K)

        # votes[n, c] = number of votes class c receives
        votes = torch.zeros(
            x_batch.size(0),
            model.num_classes,
            device=device
        )

        # --- Original OvO voting ---
        for k, (i, j) in enumerate(zip(model.pair_i.tolist(),
                                       model.pair_j.tolist())):
            pred = logits[:, k]

            votes[pred > 0, i] += 1
            votes[pred <= 0, j] += 1

        preds = votes.argmax(dim=1)

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return (correct / total)*100


def squared_hinge_loss_ovo(logits, targets, pair_i, pair_j):
    """
    Fully vectorized OvO squared hinge loss

    Args:
        logits:  (N, K)
        targets: (N,) class indices in [0, C-1]
        pair_i:  (K,)
        pair_j:  (K,)
    """
    # Expand targets to (N, K)
    t = targets.unsqueeze(1)

    # Labels: +1 for class i, -1 for class j, 0 otherwise
    y = (t == pair_i).float() - (t == pair_j).float()

    # Mask inactive classifiers
    mask = (y != 0).float()

    # Squared hinge
    margin = 1 - y * logits
    loss = torch.clamp(margin, min=0.0) ** 2

    # Apply mask and normalize
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)



def SERVER(global_model,local_models, n):
    """
    Vanilla FedAvg server: weighted average of model parameters
    """
    global_model = copy.deepcopy(global_model)

    # Zero out global params
    for param in global_model.parameters():
        param.data.zero_()

    # Weighted aggregation
    for local_model, n_i in local_models:
        weight = n_i / n
        for g_param, l_param in zip(global_model.parameters(),local_model.parameters()):
            g_param.data += weight * l_param.data

    return global_model


def client_update(model, data, lr):
    """
    Vanilla FedAvg client update using squared hinge loss
    """
    device = next(model.parameters()).device
    model = copy.deepcopy(model)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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
            loss = squared_hinge_loss_ovo(logits, y_batch.long(), model.pair_i, model.pair_j)

            loss.backward()
            optimizer.step()

    return model, n_samples


def main_fedavg(global_parameters, train_data_list, test_data, client_schedule, lr):

    global_accuracy = []

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

            local_model, n_i = client_update(global_parameters, client_data, lr)

            local_models.append((local_model, n_i))
            n += n_i

        global_parameters = SERVER(global_parameters, local_models, n)
        global_acc = test_ovo(global_parameters, test_data)
        global_accuracy.append(global_acc)

        print(f"Round {round} FedAvg Acc {global_acc:.2f}")

    # ---------------- PROFILING END ----------------
    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start
    # ------------------------------------------------
    print(f"FedAvg Total Time: {total_time:.2f} seconds")
    print(f"FedAvg Memory Used: {memory_used:.2f} MB")

    return (global_accuracy , total_time, memory_used)


import numpy as np
from config import local_epoch, device, batch_size, CR, mu
import torch, copy
from torch.utils.data import DataLoader, TensorDataset
from .fedavg import test_ovo, squared_hinge_loss_ovo
import time
import psutil
import os
import gc


def fednova_train(global_model, data, lr, momentum=0.0):
    """
    FedNova client update with squared hinge OvO
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)

    # ----- Data handling -----
    if isinstance(data, DataLoader):
        loader = data
        client_size = len(data.dataset)
    else:
        X = torch.tensor(data.drop(columns=["y"]).values, dtype=torch.float32)
        y = torch.tensor(data["y"].values, dtype=torch.long)

        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        client_size = len(X)

    tau = 0  # number of local SGD steps

    # ----- Local training -----
    for _ in range(local_epoch):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = local_model(x_batch)
            loss = squared_hinge_loss_ovo(logits, y_batch, local_model.pair_i, local_model.pair_j)
            loss.backward()
            optimizer.step()

            tau += 1

    # ----- a_i (normalization scalar) -----
    if momentum > 0:
        a_i = (tau - momentum * (1 - momentum ** tau) / (1 - momentum)) / (1 - momentum)
    else:
        a_i = tau  # no momentum case

    # ----- normalized update d_i -----
    norm_update = []
    with torch.no_grad():
        for p_g, p_l in zip(global_model.parameters(), local_model.parameters()):
            norm_update.append((p_g - p_l) / a_i)

    return norm_update, a_i, tau, client_size

def fednova_aggregate(global_model, norm_updates, a_list, client_sizes):
    """
    FedNova server aggregation (paper-correct)
    """
    total_samples = sum(client_sizes)

    # p_i = n_i / n
    p_list = [n / total_samples for n in client_sizes]

    # τ_eff = Σ p_i a_i
    tau_eff = sum(p * a for p, a in zip(p_list, a_list))

    # ----- aggregate normalized updates -----
    agg_update = []
    for param_idx in range(len(norm_updates[0])):
        s = torch.zeros_like(norm_updates[0][param_idx], device=device)
        for d_i, p_i in zip(norm_updates, p_list):
            s += p_i * d_i[param_idx]
        agg_update.append(s)

    # ----- update global model -----
    with torch.no_grad():
        for p, du in zip(global_model.parameters(), agg_update):
            p -= tau_eff * du

    return global_model


def main_fednova(global_model, train_data_list, test_data, client_schedule, lr):

    global_accuracy = []

    # ---------------- PROFILING START ----------------
    process = psutil.Process(os.getpid())
    gc.collect()

    mem_start = process.memory_info().rss / (1024**2)  # MB
    time_start = time.perf_counter()
    # --------------------------------------------------

    for round in range(CR):
        norm_updates = []
        a_list = []
        client_sizes = []

        for cid in client_schedule[round]:
            d_i, a_i, tau_i, n_i = fednova_train(global_model, train_data_list[cid], lr)
            norm_updates.append(d_i)
            a_list.append(a_i)
            client_sizes.append(n_i)

        global_model = fednova_aggregate(global_model, norm_updates, a_list, client_sizes)
        global_acc = test_ovo(global_model, test_data)
        global_accuracy.append(global_acc)

        print(f"Round {round} FedNova Acc {global_acc:.2f}")

    # ---------------- PROFILING END ----------------
    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start
    # ------------------------------------------------

    return (global_accuracy , total_time, memory_used)


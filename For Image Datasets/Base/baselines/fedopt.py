import numpy as np
from config import local_epoch, batch_size, CR 
import torch, copy
from torch.utils.data import DataLoader, TensorDataset
from .fedavg import test_ovo, squared_hinge_loss_ovo
import time
import psutil
import os
import gc


def init_fedopt_state(model):
    model_device = next(model.parameters()).device
    return {
        "m": [torch.zeros_like(p, device=model_device) for p in model.parameters()],
        "v": [torch.zeros_like(p, device=model_device) for p in model.parameters()],
        "t": 0
    }

def server_fedadam(global_model, local_deltas, fedopt_state,
                   server_lr=1, beta1=0.9, beta2=0.999, eps=1e-8):

    total_samples = sum(n_k for _, n_k in local_deltas)
    global_params = list(global_model.parameters())
    model_device = next(global_model.parameters()).device

    # 1. Aggregate Δw
    delta_t = []
    for p_idx, p in enumerate(global_params):
        d = torch.zeros_like(p, device=model_device)
        for del_w_k, n_k in local_deltas:
            pk = n_k / total_samples
            d += pk * del_w_k[p_idx].to(model_device)
        delta_t.append(d)

    # 2. FedAdam update
    fedopt_state["t"] += 1
    t = fedopt_state["t"]

    with torch.no_grad():
        for i, (p, d) in enumerate(zip(global_params, delta_t)):
            fedopt_state["m"][i] = beta1 * fedopt_state["m"][i] + (1 - beta1) * d
            fedopt_state["v"][i] = beta2 * fedopt_state["v"][i] + (1 - beta2) * (d ** 2)

            m_hat = fedopt_state["m"][i] / (1 - beta1 ** t)
            v_hat = fedopt_state["v"][i] / (1 - beta2 ** t)

            p.add_(server_lr * m_hat / (torch.sqrt(v_hat) + eps))

    return global_model



def client_update_fedopt(global_model, data, lr):
    """
    FedOpt client update.
    Returns model delta Δw_k = w_k - w_global
    """
    device = next(global_model.parameters()).device

    # Copy global model
    model = copy.deepcopy(global_model).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # -------- Data handling --------
    if isinstance(data, DataLoader):
        loader = data
        n_samples = len(data.dataset)
    else:
        X = torch.tensor(data.drop(columns=["y"]).values,dtype=torch.float32)
        y = torch.tensor(data["y"].values,dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y),batch_size=batch_size,shuffle=True)
        n_samples = len(X)

    # -------- Local training --------
    for _ in range(local_epoch):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)

            loss = squared_hinge_loss_ovo(
                logits, y_batch, model.pair_i, model.pair_j
            )

            loss.backward()
            optimizer.step()

    # -------- Compute Δw_k --------
    del_w = []
    with torch.no_grad():
        for p_k, p_g in zip(model.parameters(),
                            global_model.parameters()): 
            del_w.append((p_k - p_g).detach().cpu())

    return del_w, n_samples


def fedopt_main(global_model, train_data_list, test_data, client_schedule, lr, server_lr=0.1, beta1=0.9, beta2=0.999, eps=1e-3):

    # ---------------- PROFILING START ----------------
    process = psutil.Process(os.getpid())
    gc.collect()

    mem_start = process.memory_info().rss / (1024**2)  # MB
    time_start = time.perf_counter()
    # --------------------------------------------------
    global_accuracy = []
    fedopt_state = init_fedopt_state(global_model)

    for r in range(CR):
        local_deltas = []

        for cid in client_schedule[r]:
            del_w_k, n_k = client_update_fedopt(global_model, train_data_list[cid], lr)
            local_deltas.append((del_w_k, n_k))

        global_model = server_fedadam(global_model, local_deltas, fedopt_state, server_lr=server_lr, beta1=beta1, beta2=beta2, eps=eps)

        acc = test_ovo(global_model, test_data)
        global_accuracy.append(acc)
        print(f"[FedOpt] Round {r+1}/{CR} | Acc {acc:.4f}")

    # ---------------- PROFILING END ----------------
    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start
    # ------------------------------------------------

    return (global_accuracy , total_time, memory_used)


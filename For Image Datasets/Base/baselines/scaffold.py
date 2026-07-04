import numpy as np
from config import local_epoch, device, batch_size, CR, num_clients
import torch, copy
from torch.utils.data import DataLoader, TensorDataset
from .fedavg import test_ovo, squared_hinge_loss_ovo
import time
import psutil
import os
import gc

def init_control_variates(model):
    return [torch.zeros_like(p) for p in model.parameters()]


def server_update_scaffold(global_model, client_delta_ys, client_delta_cs, c_global, eta_g, num_clients):
    """
    Paper-correct SCAFFOLD server update
    """
    S = len(client_delta_ys)

    # ----- Aggregate Δx -----
    agg_delta_x = []
    for param_idx in range(len(client_delta_ys[0])):
        s = torch.zeros_like(client_delta_ys[0][param_idx])
        for d in client_delta_ys:
            s += d[param_idx]
        agg_delta_x.append(s / S)

    # ----- Update global model -----
    with torch.no_grad():
        for p, dx in zip(global_model.parameters(), agg_delta_x):
            p += eta_g * dx

    # ----- Aggregate Δc -----
    agg_delta_c = []
    for param_idx in range(len(client_delta_cs[0])):
        s = torch.zeros_like(c_global[param_idx], device=device)
        for dc in client_delta_cs:
            s += dc[param_idx]
        agg_delta_c.append(s / S)

    # ----- Update global control variate -----
    scale = S / num_clients
    new_c_global = []
    for cg, dc in zip(c_global, agg_delta_c):
        new_c_global.append(cg + scale * dc)

    return global_model, new_c_global


def client_update_scaffold(global_model, data, c_global, c_local, lr):
    """
    Paper-correct SCAFFOLD client update
    """
    device = next(global_model.parameters()).device
    model = copy.deepcopy(global_model).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # ----- Data -----
    if isinstance(data, DataLoader):
        loader = data
        n_samples = len(data.dataset)
    else:
        X = torch.tensor(data.drop(columns=["y"]).values, dtype=torch.float32)
        y = torch.tensor(data["y"].values, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        n_samples = len(X)

    # Snapshot x, notheing but initial global model parameters
    x_init = [p.detach().clone() for p in global_model.parameters()]

    steps = 0

    # ----- Local SGD -----
    for _ in range(local_epoch):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = squared_hinge_loss_ovo(logits, y_batch, model.pair_i, model.pair_j)
            loss.backward()

            # g - c_i + c
            with torch.no_grad():
                for p, cg, cl in zip(model.parameters(), c_global, c_local):
                    p.grad += cg - cl

            optimizer.step()
            steps += 1

    # ----- Δy = y_i - x -----
    delta_y = []
    for p, p0 in zip(model.parameters(), x_init):
        delta_y.append((p.detach() - p0).clone())

    # ----- c_i update (Option II) -----
    new_c_local = []
    delta_c = []

    for p, p0, cl, cg in zip(model.parameters(),x_init,c_local,c_global):
        with torch.no_grad():
            ci_new = cl - cg + (p0 - p.detach()) / (steps * lr)
            new_c_local.append(ci_new.detach())
            delta_c.append((ci_new - cl).detach())

    return delta_y, delta_c, new_c_local, n_samples




def main_scaffold(global_parameters, train_data_list, test_data, client_schedule, lr):
    # ---------------- PROFILING START ----------------
    process = psutil.Process(os.getpid())
    gc.collect()

    mem_start = process.memory_info().rss / (1024**2)  # MB
    time_start = time.perf_counter()
    # --------------------------------------------------
    global_accuracy = []

    c_global = init_control_variates(global_parameters)
    c_local = {i: init_control_variates(global_parameters) for i in range(num_clients)}

    for round in range(CR):
        client_delta_ys = []
        client_delta_cs = []
        selected_clients = client_schedule[round]

        for cid in selected_clients:
            dy, dc, new_c, _ = client_update_scaffold(global_parameters,train_data_list[cid],c_global,c_local[cid], lr)
            c_local[cid] = new_c
            client_delta_ys.append(dy)
            client_delta_cs.append(dc)

        global_parameters, c_global = server_update_scaffold(global_parameters,client_delta_ys,client_delta_cs,c_global,eta_g=1.0,num_clients=num_clients)

        global_acc = test_ovo(global_parameters, test_data)
        global_accuracy.append(global_acc)

        print(f"Round {round} SCAFFOLD Acc {global_acc:.2f}")

    # ---------------- PROFILING END ----------------
    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start
    # ------------------------------------------------

    return (global_accuracy , total_time, memory_used)



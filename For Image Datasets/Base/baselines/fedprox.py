import numpy as np
from config import local_epoch, device, batch_size, CR, mu
import torch, copy
from torch.utils.data import DataLoader, TensorDataset
from .fedavg import test_ovo, SERVER, squared_hinge_loss_ovo
import time
import psutil
import os
import gc


def client_update_fedprox(global_model, data, lr):
    """
    FedProx client update (minimal modification of FedAvg)

    Adds proximal term:
        (mu / 2) * ||w - w_global||^2
    """
    model = copy.deepcopy(global_model)         # local model
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # --- Data handling (same as FedAvg) ---
    if isinstance(data, DataLoader):
        loader = data
        n_samples = len(data.dataset)
    else: 
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

            logits = model(x_batch)
            loss = squared_hinge_loss_ovo(logits, y_batch, model.pair_i, model.pair_j)

            # -------- FedProx proximal term --------
            prox = 0.0
            for w, w_global in zip(model.parameters(), global_model.parameters()):
                prox += torch.norm(w - w_global) ** 2

            loss = loss + (mu / 2) * prox
            # --------------------------------------

            loss.backward()
            optimizer.step()

    return model, n_samples


def main_fedprox(global_parameters, train_data_list, test_data, client_schedule, lr):

    # ---------------- PROFILING START ----------------
    process = psutil.Process(os.getpid())
    gc.collect()

    mem_start = process.memory_info().rss / (1024**2)  # MB
    time_start = time.perf_counter()
    # --------------------------------------------------
    global_accuracy = []

    for round in range(CR):

        client_set = client_schedule[round]
        local_models = []
        n = 0

        for client in client_set:
            client_data = train_data_list[client]

            local_model, n_i = client_update_fedprox(global_parameters, client_data, lr)

            local_models.append((local_model, n_i))
            n += n_i

        global_parameters = SERVER(global_parameters, local_models, n)
        global_acc = test_ovo(global_parameters, test_data)
        global_accuracy.append(global_acc)

        print(f"[FedProx] Round {round} Acc {global_acc:.2f}")

    # ---------------- PROFILING END ----------------
    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start
    # ------------------------------------------------

    return (global_accuracy , total_time, memory_used)


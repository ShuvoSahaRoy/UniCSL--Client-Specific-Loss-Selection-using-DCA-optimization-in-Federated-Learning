import numpy as np
from config import local_epoch, device, batch_size, CR, alpha_coef, num_clients
import torch, copy
from torch.utils.data import DataLoader, TensorDataset
import time
import psutil
import os
import gc
from .loss_functions import get_loss_fn
from .feddyn import get_mdl_params, set_mdl_params, server_update_feddyn
from .fedavg_multi_loss import test_ovo, loss_functions_list

def train_feddyn(global_model, data, h_k, theta_prev, alpha_k, current_lr, loss_name, loss_params):
    model = copy.deepcopy(global_model).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, weight_decay= alpha_k* 1e-3)
    loss_fn = get_loss_fn(loss_name)

    if isinstance(data, DataLoader):
        loader = data
    else:
        X = torch.tensor(data.drop(columns=["y"]).values, dtype=torch.float32)
        y = torch.tensor(data["y"].values, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    theta_prev = theta_prev.to(device)
    h_k = h_k.to(device)
    linear_penalty_coef = alpha_k * (h_k - theta_prev)

    for _ in range(local_epoch):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            logits = model(x_batch)

            if loss_name in ['Least Squares', 'Squared Hinge']:
                loss_task = loss_fn(logits, y_batch.long(), model.pair_i, model.pair_j)
            else:
                loss_task = loss_fn(logits, y_batch.long(), model.pair_i, model.pair_j, *loss_params)

            theta = torch.cat([p.view(-1) for p in model.parameters()])
            linear_penalty = torch.sum(theta * linear_penalty_coef)
            loss = loss_task + linear_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    theta_new = get_mdl_params(model).detach()
    new_h_k = h_k + (theta_new - theta_prev)

    return theta_new.cpu(), new_h_k.cpu(), model

def client_update(global_model, data, h_k, theta_prev, alpha_k, current_lr, loss_dict, val_data):
    if loss_dict['best_loss'] is None:
        best_f1 = -1
        best_theta_new = None
        best_new_h_k = None
        best_loss = None
        
        for loss_cfg in loss_functions_list:
            loss_name = list(loss_cfg.keys())[0]
            loss_params = loss_cfg[loss_name]
            
            theta_new, new_h_k, temp_model = train_feddyn(
                global_model, data, h_k, theta_prev, alpha_k, current_lr, loss_name, loss_params
            )
            
            accuracy, f1 = test_ovo(temp_model, val_data)
            
            if f1 > best_f1:
                best_f1 = f1
                best_theta_new = theta_new
                best_new_h_k = new_h_k
                best_loss = loss_cfg
                
        return best_theta_new, best_new_h_k, best_loss
    else:
        best_loss = loss_dict['best_loss']
        loss_name = list(best_loss.keys())[0]
        loss_params = best_loss[loss_name]
        
        theta_new, new_h_k, _ = train_feddyn(
            global_model, data, h_k, theta_prev, alpha_k, current_lr, loss_name, loss_params
        )
        return theta_new, new_h_k, best_loss

def main_feddyn_multi_loss(global_model, train_data_list, val_data_list, test_data, client_schedule, lr):
    process = psutil.Process(os.getpid())
    gc.collect()

    mem_start = process.memory_info().rss / (1024**2)
    time_start = time.perf_counter()

    current_lr = lr
    n_par = sum(p.numel() for p in global_model.parameters())

    theta_global = get_mdl_params(global_model)
    clnt_thetas = torch.stack([theta_global.clone() for _ in range(num_clients)])
    h_locals = torch.zeros((num_clients, n_par))

    sample_counts = np.array([len(d) for d in train_data_list])
    weight_list = sample_counts / np.sum(sample_counts) * num_clients

    global_accuracy = []
    loss_dictionary = {client_id: {"best_loss": None} for client_id in range(num_clients)}

    for r in range(CR):
        selected = client_schedule[r]
        theta_prev = theta_global.clone()

        for k in selected:
            alpha_k = alpha_coef / weight_list[k]
            
            val_data = val_data_list[k]

            theta_k, h_k, best_loss = client_update(
                global_model, 
                train_data_list[k], 
                h_locals[k],
                theta_prev,
                alpha_k,
                current_lr,
                loss_dictionary[k],
                val_data
            )

            clnt_thetas[k] = theta_k
            h_locals[k] = h_k
            loss_dictionary[k]['best_loss'] = best_loss

        theta_global = server_update_feddyn(clnt_thetas[selected], h_locals)
        global_model = set_mdl_params(global_model, theta_global)

        acc, f1 = test_ovo(global_model, test_data)
        global_accuracy.append(acc)
        print(f"[FedDyn Multi-Loss] Round {r+1}/{CR} | Acc {acc:.2f}%")

    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start

    print(f"FedDyn Multi-Loss Total Time: {total_time:.2f} seconds")
    print(f"FedDyn Multi-Loss Memory Used: {memory_used:.2f} MB")

    return (global_accuracy, total_time, memory_used), loss_dictionary

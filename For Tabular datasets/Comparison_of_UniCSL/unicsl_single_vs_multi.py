# this unicsl v1 slect the best loss function for each client and then train the model and keep it static

import numpy as np
from unisvm import train_with_covergence, test 
from config import num_clients, participants, CR, local_epoch, non_iid
import time
import psutil
import os
import gc


def SERVER(local_updates, n):
    w = []
    for update in local_updates:
        w.append(update[0] * (update[1]/n))
    
    w = np.array(w)
    
    # global_model = g_model - lr * np.sum(w,axis=0)
    global_model = np.sum(w,axis=0)

    # global_model = global_model/np.linalg.norm(global_model) 
    return global_model


loss_functions = [
    {'Least Squares': [2]},
    {'Smooth Hinge': [10]},
    {'Squared Hinge': [0]},
    {'Truncated SH': [2]},
    {'Truncated LS': [2]},
    {'Smoothed Ramp1': [2]},
    {'Smoothed Ramp2': [2, 10]},
    {'nonconvex exp loss1': [2, 2, 3]},
    {'nonconvex exp loss2': [2, 2, 4]},
    {'nonconvex exp loss3': [2, 3, 4]},
    {'Huber loss': [1]}
                ]


# global_parameters.copy(), client_data, test_val_data[1], client_Q_QXT[client],loss_dictionary[client]
def client_update(global_weights, data, val_data, client_Q_QXT, loss):
    if loss['best_loss'] is None:
        best_f1 = -1
        best_w_f1 = None
        best_loss_f1 = None

        best_iter = local_epoch + 1
        best_w_iter = None
        best_loss_iter = None

        all_iters = []

        for loss_fn in loss_functions:
            w, no_of_sample, client_iter = train_with_covergence(global_weights.copy(), data, client_Q_QXT, loss_fn)
            all_iters.append(client_iter)

            # Track best F1 (in case we fall back)
            accuracy, f1 = test(w, val_data)
            if f1 > best_f1:
                best_f1 = f1
                best_w_f1 = w
                best_loss_f1 = loss_fn
                best_f1_iter = client_iter  # save in case needed

            # Track best convergence
            if client_iter < best_iter:
                best_iter = client_iter
                best_w_iter = w
                best_loss_iter = loss_fn

        # Decide based on convergence
        if all(i >= local_epoch for i in all_iters):
            # No convergence: fall back to highest F1
            selected_w = best_w_f1
            selected_loss = best_loss_f1
            selected_iter = best_f1_iter
        else:
            # At least one converged: pick lowest iteration
            selected_w = best_w_iter
            selected_loss = best_loss_iter
            selected_iter = best_iter

        # del_w = global_weights - selected_w
        del_w = selected_w

        return (del_w, no_of_sample, best_f1_iter), best_loss_f1
    else:
        best_loss = loss['best_loss']
        w, no_of_sample, client_iter = train_with_covergence(global_weights.copy(), data, client_Q_QXT, best_loss)
        # del_w = global_weights - w
        del_w = w
        
    return (del_w, no_of_sample, client_iter), best_loss


def main_unicsl(global_parameters, client_schedule, train_data_list, test_val_data, clients_Q_QXT, loss = None):

    if loss == None:
        loss_dictionary = {client_id: {"best_loss": None} for client_id in range(num_clients)}
    else:
        loss_dictionary = {client_id: {"best_loss": loss} for client_id in range(num_clients)}

    global_accuracy = []
    total_iteration = 0
    t_i_p_c = [] # Total iterations per client
    client_iter_data = np.zeros((num_clients, CR), dtype=int)

    # ---------------- PROFILING START ----------------
    process = psutil.Process(os.getpid())
    gc.collect()

    mem_start = process.memory_info().rss / (1024**2)  # MB
    time_start = time.perf_counter()
    # --------------------------------------------------

    for round in range(CR):

        # select random clients
        client_set = client_schedule[round]

        n = 0
        local_updates = []
        total_iteration_per_cr = 0

        for client in client_set:
            client_data = train_data_list[client]

            local_update, loss = client_update(global_parameters.copy(), client_data, test_val_data[1], clients_Q_QXT[client],loss_dictionary[client])
            
            loss_dictionary[client]['best_loss'] = loss
            local_updates.append(local_update)
            n = n + local_update[1]
            total_iteration_per_cr += local_update[2]
            client_iter_data[client, round] = local_update[2] 

        total_iteration += total_iteration_per_cr
        t_i_p_c.append(total_iteration_per_cr)
        global_parameters = SERVER(local_updates, n)
        # print(f"global_parameters {global_parameters}")

        global_acc, _ = test(global_parameters,test_val_data[0])
        global_accuracy.append(global_acc)

        print(f" {'-' * 10} communication round {round+1} {'-' * 10}")
        print(f"{10 * '-'} Global accuracy {global_acc:.4f}{10 * '-'}\n")

    # ---------------- PROFILING END ----------------
    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start
    # ------------------------------------------------

    total_iteration = np.sum(t_i_p_c)
    return (global_accuracy, total_iteration, client_iter_data, loss_dictionary, total_time, memory_used)
# this unicsl v1 slect the best loss function for each client and then train the model and keep it static
import time
import tracemalloc
import numpy as np
from .utils.unisvm import train, test
from config import num_clients, CR, SEED, non_iid, corrupt_data
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
    global_model = global_model / np.linalg.norm(global_model)  # Normalize the global model

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
    {'nonconvex exp loss3': [2, 3, 4]}
                ]


# def calculate_inverse(train_data_list):

#     clients_Q_QXT = []
#     for client in range(num_clients):
#         X = train_data_list[client].iloc[:, :-1]
#         I = np.eye(X.shape[1]).astype('int64')
#         c = 10e-1
#         Q = np.linalg.inv((c * I) + np.dot(X.T, X))
#         Q_X_t = np.dot(Q, X.T)

#         clients_Q_QXT.append((Q_X_t))
    
#     return clients_Q_QXT


import itertools

def calculate_inverse_ovo(train_data_list, num_classes,c):

    clients_Q_QXT = {}
    pairs = list(itertools.combinations(range(num_classes), 2))
    
    for client in range(num_clients):
        clients_Q_QXT[client] = {}
        X_all = train_data_list[client].iloc[:, :-1].values
        y_all = train_data_list[client].iloc[:, -1].values
        
        for k, (i, j) in enumerate(pairs):
            # Isolate data for the specific pair
            mask = np.logical_or(y_all == i, y_all == j)
            X_pair = X_all[mask]
            
            # Handle edge case: client lacks data for this specific pair
            if len(X_pair) == 0:
                clients_Q_QXT[client][k] = None
                continue
                
            I = np.eye(X_pair.shape[1]).astype('int64')
            Q = np.linalg.inv((c * I) + np.dot(X_pair.T, X_pair))
            Q_X_t = np.dot(Q, X_pair.T)
            
            clients_Q_QXT[client][k] = Q_X_t
            
    return clients_Q_QXT


# global_parameters.copy(), client_data, test_data[1], client_Q_QXT[client],loss_dictionary[client]
def client_update(global_weights, data, client_Q_QXT, loss,val_data):
    if loss['best_loss'] is None:
        best_f1 = -1
        # best_accuracy = -1
        for loss in loss_functions:
            w, no_of_sample = train(global_weights.copy(), data, client_Q_QXT, loss)
            accuracy, f1 = test(w, val_data)

            if f1 > best_f1: 
                best_f1 = f1
                best_w = w
                best_loss = loss

            # if accuracy > best_accuracy:
            #     best_accuracy = accuracy
            #     best_w = w
            #     best_loss = loss

        # del_w = global_weights - best_w
        local_weights = best_w

    else:
        best_loss = loss['best_loss']
        local_weights, no_of_sample = train(global_weights.copy(), data, client_Q_QXT, best_loss)
        # del_w = global_weights - w
        
    return (local_weights, no_of_sample), best_loss


def main_unicsl(global_model, train_data_list,val_data_list, test_data, client_schedule, num_classes, c):
    print(f"SEED: {SEED}, non_iid: {non_iid}, corrupt_data: {corrupt_data}, C: {c}")
    loss_dictionary = {client_id: {"best_loss": None} for client_id in range(num_clients)}

    # if dataset_name == 'mnist':
    #     loss_dictionary = {client_id: {"best_loss": {'Truncated SH': [2]}} for client_id in range(num_clients)}
    # elif dataset_name == 'fashionmnist':
    #     loss_dictionary = {client_id: {"best_loss": {'Smoothed Ramp1': [2]},} for client_id in range(num_clients)}
    # else:
    #     loss_dictionary = {client_id: {"best_loss": {'Truncated SH': [2]}} for client_id in range(num_clients)}

    global_accuracy = []

    # features = 784
    # num_pairs = int(10 * (10 - 1) / 2)
    
    # # Initialize correctly for OVO: (features, num_pairs)
    # global_parameters = np.random.rand(features, num_pairs) 
    global_parameters = global_model.linear.weight.data.cpu().numpy().T
   
    Q_process = psutil.Process(os.getpid())
    gc.collect()
    mem_start_Q = Q_process.memory_info().rss / (1024**2)  # MB
    # Precalculate matrices pairwise
    inv_time_start = time.perf_counter()

    client_Q_QXT = calculate_inverse_ovo(train_data_list, num_classes, c)

    total_time_inv = time.perf_counter() - inv_time_start
    mem_end_Q = Q_process.memory_info().rss / (1024**2)  # MB
    total_memory_Q = mem_end_Q - mem_start_Q

   
    # client_Q_QXT = calculate_inverse(train_data_list)  # works for binary

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

        for client in client_set:
            client_data = train_data_list[client]

            local_update, loss = client_update(global_parameters.copy(), client_data, client_Q_QXT[client],loss_dictionary[client],val_data_list[client])

            loss_dictionary[client]['best_loss'] = loss
            local_updates.append(local_update)
            n = n + local_update[1]

        global_parameters = SERVER(local_updates, n)

        global_acc, _ = test(global_parameters,test_data)

        global_accuracy.append(global_acc)

 
        print(f"round {round+1}:UniCSL Global accuracy {global_acc:.2f}")

    # ---------------- PROFILING END ----------------
    time_end = time.perf_counter()
    mem_end = process.memory_info().rss / (1024**2)

    total_time = time_end - time_start
    memory_used = mem_end - mem_start

    print(f"Inverse calculation time: {total_time_inv}")
    print(f"Memory used for Q_QXT: {total_memory_Q} MB")
    print(f"\nMemory used for training: {memory_used} MB, time used for training: {total_time} seconds\n")
    # ------------------------------------------------

    return (global_accuracy , total_time+total_time_inv, memory_used+total_memory_Q), loss_dictionary
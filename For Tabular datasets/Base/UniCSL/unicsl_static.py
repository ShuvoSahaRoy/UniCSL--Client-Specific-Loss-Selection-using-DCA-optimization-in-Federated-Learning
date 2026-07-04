# this unicsl v1 slect the best loss function for each client and then train the model and keep it static

import tracemalloc
import numpy as np
from .utils.unisvm import train, test 
from config import num_clients, CR
import time

def SERVER(g_model, local_updates, n):
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
    {'nonconvex exp loss3': [2, 3, 4]}
                ]


def calculate_inverse(train_data_list):

    clients_Q_QXT = []
    for client in range(num_clients):
        X = train_data_list[client].iloc[:, :-1]
        I = np.eye(X.shape[1]).astype('int64')
        c = 10e-8
        Q = np.linalg.inv((c * I) + np.dot(X.T, X))
        Q_X_t = np.dot(Q, X.T)

        clients_Q_QXT.append((Q_X_t))
    
    return clients_Q_QXT


# global_parameters.copy(), client_data, test_val_data[1], client_Q_QXT[client],loss_dictionary[client]
def client_update(global_weights, data, val_data, client_Q_QXT, loss):
    if loss['best_loss'] is None:
        best_f1 = -1
        for loss in loss_functions:
            w, no_of_sample = train(global_weights.copy(), data, client_Q_QXT, loss)
            accuracy, f1 = test(w, val_data)
            # print(f"loss function {loss} and f1 {f1} and accuracy {accuracy:.4f}\n")
            if f1 > best_f1:
                best_f1 = f1
                best_w = w
                best_loss = loss
        # print(f"loss function {best_loss} and f1 {best_f1}\n")
        # del_w = global_weights - best_w
        del_w = best_w

    else:
        best_loss = loss['best_loss']
        w, no_of_sample = train(global_weights.copy(), data, client_Q_QXT, best_loss)
        # del_w = global_weights - w
        del_w = w
    
    
    return (del_w, no_of_sample), best_loss


def main_unicsl(global_parameters, client_schedule, train_data_list, test_val_data):

    loss_dictionary = {client_id: {"best_loss": None} for client_id in range(num_clients)}

    global_parameters = global_parameters[:-1]
    global_accuracy = []


    start_time = time.time()
    tracemalloc.start()
    # print(f"global_parameters initialization {global_parameters}")
    # this line of code is written to reduce the overhead calculation time of Q and Q_X_t for each client, beacuse it's repeated in selection of clients
    """This line doesn't represent anything on serverside, this just for the sake of make simulation faster"""
    client_Q_QXT = calculate_inverse(train_data_list)

    for round in range(CR):

        # select random clients
        client_set = client_schedule[round]

        n = 0
        local_updates = []

        for client in client_set:
            client_data = train_data_list[client]

            # print(f"client {client} data shape {client_data.shape}")
            local_update, loss = client_update(global_parameters.copy(), client_data, test_val_data[1], client_Q_QXT[client],loss_dictionary[client])
            
            loss_dictionary[client]['best_loss'] = loss
            local_updates.append(local_update)
            n = n + local_update[1]

        global_parameters = SERVER(global_parameters,local_updates, n)

        # print(f"global_parameters {global_parameters}")
        # time.sleep(10)
        global_acc, _ = test(global_parameters,test_val_data[0])

        global_accuracy.append(global_acc)
        print(f"[UniCSL Static] Round {round + 1}/{CR} | Acc: {global_acc:.2f}")


    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    total_time = end_time - start_time

    return (global_accuracy , total_time, (peak_memory / (1024**2))), loss_dictionary
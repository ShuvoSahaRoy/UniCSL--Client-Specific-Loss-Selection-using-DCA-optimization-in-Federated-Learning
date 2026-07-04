# this unicsl v1 slect the best loss function for each client in each round (dynamic in each round)

import numpy as np
from .utils.unisvm import train, test 
from config import num_clients, participants, CR, alpha, non_iid, min_samples_per_client


def SERVER(g_model, local_updates, lr, n):
    w = []
    for update in local_updates:
        w.append(update[0] * (update[1]/n))
    
    w = np.array(w)
    
    global_model = g_model - lr * np.sum(w,axis=0)
    # global_model = np.sum(w,axis=0)

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
def client_update(global_weights, data, val_data, client_Q_QXT):
    best_f1 = 0
    for loss in loss_functions:
        w, no_of_sample = train(global_weights.copy(), data, client_Q_QXT, loss)
        accuracy, f1 = test(w, val_data)
        # print(f"loss function {loss} and f1 {f1} and accuracy {accuracy:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_w = w
            best_loss = loss
    # print(f"loss function {best_loss} and f1 {best_f1}")
    del_w = global_weights - best_w
        
    return (del_w, no_of_sample), best_loss


def main_unicsl(train_data_list, test_val_data, lr, dataset):
    message = f'UniCSL NonIID - {non_iid}, lr {lr} participants {participants} num clients {num_clients} alpha {alpha} min_samples_per_client {min_samples_per_client}'

    features = train_data_list[0].shape[1] - 1

    global_parameters = np.random.rand(features)
    global_accuracy = []
    # print(f"global_parameters initialization {global_parameters}")
    # this line of code is written to reduce the overhead calculation time of Q and Q_X_t for each client, beacuse it's repeated in selection of clients
    """This line doesn't represent anything on serverside, this just for the sake of make simulation faster"""
    client_Q_QXT = calculate_inverse(train_data_list)

    for round in range(CR):

        # select random clients
        client_set = np.random.choice(num_clients, size=int(participants), replace=False)

        n = 0
        local_updates = []

        for client in client_set:
            client_data = train_data_list[client]

            local_update, loss = client_update(global_parameters.copy(), client_data, test_val_data[1], client_Q_QXT[client])
            
            local_updates.append(local_update)
            n = n + local_update[1]

        global_parameters = SERVER(global_parameters,local_updates, lr, n)
        # print(f"global_parameters {global_parameters}")

        global_acc, _ = test(global_parameters,test_val_data[0])

        global_accuracy.append(global_acc)
        if (round+1) % 10 == 0:
            print(f" {'-' * 10} communication round {round+1} {'-' * 10}")
            print(f"{10 * '-'}{dataset} Global accuracy {global_acc:.4f}{10 * '-'}\n")


        # if convergence_check:
        #     if check_convergence(global_accuracy, window_size=20, tolerance=0.01):
        #         lr *= 0.9  # Reduce learning rate
    # save_results_to_excel("results/accuracy_results.xlsx", dataset.split('.')[0], message, global_accuracy)
    # plot_and_write_function_frequencies(loss_dictionary, dataset, excel_file=f'results/loss_frequencies {non_iid}_{lr}.xlsx')

    return global_accuracy
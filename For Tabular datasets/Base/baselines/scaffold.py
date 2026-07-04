import tracemalloc
import numpy as np
from config import num_clients, participants, CR, local_epoch
from Base.classifier import logistic_regression
from Base.utils.utils import SERVER
import time, tracemalloc
### this code is designed based on "Federated learning on Non-IID data silos: an experimental Study"

def client_update(model, data, control_variate, global_control_variates, lr):
    """
    return : del_w, del_c, number_of_sample, updated_client_control_variates
    """
    global_model = model.copy()

    number_of_sample = data.shape[0]
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    for i in range(local_epoch):
        w = model[:-1]
        b = model[-1]
        linear = np.dot(X, w) + b
        # apply sigmoid function
        y_pred = 1 / (1 + np.exp(-linear))
        
        # gradient of binary cross entropy
        dw = (1 / number_of_sample) * np.dot(X.T, (y_pred - y))
        db = (1 / number_of_sample) * np.sum(y_pred - y)
        
        gradient = np.concatenate((dw,[db]))

        corrected_gradient = gradient - control_variate + global_control_variates
        model = model - lr * corrected_gradient

    del_w = global_model - model

    # Update the local control variate
    new_control_variate = control_variate - global_control_variates + (global_model - model)* (1/(lr*local_epoch))
    del_c = new_control_variate - control_variate

    return del_w, del_c, new_control_variate

# global_parameters, client_updates, lr, n, client_control_updates, global_control_variates
def SERVER(global_model, local_updates,client_control_updates, global_control_variates, server_lr=1.0):

    delta_x = np.mean([w for w, _ in local_updates], axis=0)
    global_model = global_model - server_lr * delta_x

    mean_delta_c = np.mean(client_control_updates, axis=0)
    # Update the global control variate
    global_control_variates = global_control_variates + mean_delta_c * (participants / num_clients)
    
    return global_model, global_control_variates


def main_scaffold(global_parameters, client_schedule, train_data_list, test_data, lr):

    features = train_data_list[0].shape[1]
    client_control_variates = np.zeros((num_clients,features))
    global_control_variates = np.zeros_like(global_parameters)

    global_accuracy = []

    start_time = time.time()
    tracemalloc.start()
    for round in range(CR):

        client_set = client_schedule[round]
        n = 0
        client_updates = []
        client_control_updates = []

        for client in client_set:
            client_data = train_data_list[client]

            del_w,  del_c, new_c_client = client_update(global_parameters.copy(), client_data, client_control_variates[client], global_control_variates, lr)

            client_control_variates[client] = new_c_client

            client_updates.append((del_w,client_data.shape[0]))
            client_control_updates.append(del_c)
            n = n + client_data.shape[0]


        global_parameters, global_control_variates = SERVER(global_parameters, client_updates,
                                                            client_control_updates, global_control_variates)

        global_acc = logistic_regression.test(global_parameters, test_data)
        global_accuracy.append(global_acc)
        print(f"[SCAFFOLD] Round {round + 1}/{CR} | Acc: {global_acc:.2f}")

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time
 
    return (global_accuracy , total_time, (peak_memory / (1024**2)))
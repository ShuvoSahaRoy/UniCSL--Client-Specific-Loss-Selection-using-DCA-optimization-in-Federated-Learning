import numpy as np
from config import *
from Base.classifier import logistic_regression
from Base.classifier import svm
from Base.utils.utils import SERVER
import tracemalloc
import time

def client_update(model, data, lr):
    global_model = model.copy()

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    w = model[:-1]
    b = model[-1]
    
    number_of_sample = X.shape[0]
    
    for i in range(local_epoch):
        linear = np.dot(X, w) + b
        y_pred = 1 / (1 + np.exp(-linear))
        
        # gradient of binary cross entropy
        dw = (1 / number_of_sample) * np.dot(X.T, (y_pred - y))
        db = (1 / number_of_sample) * np.sum(y_pred - y)
        
        # Proximal term added to the gradients (FedProx modification)
        dw += mu * (w - global_model[:-1])
        db += mu * (b - global_model[-1])

        w = w - lr * dw
        b = b - lr * db 
    
    model = np.concatenate((w, [b]))
    
    # del_w = global_model - model
    del_w = model
    local_update = (del_w, number_of_sample)
    return local_update

def main_fedprox(global_parameters, client_schedule, train_data_list, test_data, lr):

    global_accuracy = []
    start_time = time.time()
    tracemalloc.start()

    for round in range(CR):

        client_set = client_schedule[round]
        n = 0
        local_updates = []

        for client in client_set:
            client_data = train_data_list[client]

            local_update = client_update(global_parameters.copy(), client_data, lr)

            local_updates.append(local_update)
            n = n + client_data.shape[0]

        global_parameters = SERVER(global_parameters,local_updates, n)

        global_acc = logistic_regression.test(global_parameters,test_data)
        global_accuracy.append(global_acc)
        print(f"[FedProx] Round {round + 1}/{CR} | Acc: {global_acc:.2f}")

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time
 
    return (global_accuracy , total_time, (peak_memory / (1024**2)))
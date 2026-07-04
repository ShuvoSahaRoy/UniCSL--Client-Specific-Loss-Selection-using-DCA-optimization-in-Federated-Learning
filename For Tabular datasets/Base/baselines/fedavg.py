import numpy as np
from config import *
from Base.classifier import logistic_regression
from Base.utils.utils import SERVER
import tracemalloc
import time

def client_update(model, data, lr):
    global_model = model.copy()
  
    model = logistic_regression.train(model, data, lr)

    # del_w = global_model - model

    return (model, data.shape[0])


def main_fedavg(global_parameters, client_schedule, train_data_list, test_data, lr):

    global_accuracy = []

    start_time = time.time()
    tracemalloc.start()

    for round in range(CR):
        # select random clients
        client_set = client_schedule[round]
        # print(f"Round {round + 1}/{CR}, Selected Clients: {client_set}")
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
        print(f"[FedAvg] Round {round + 1}/{CR} | Acc: {global_acc:.2f}")
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time
 
    return (global_accuracy , total_time, (peak_memory / (1024**2)))
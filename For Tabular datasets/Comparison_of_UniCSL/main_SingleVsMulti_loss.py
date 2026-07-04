import pandas as pd
# Add project root to Python path
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *                       
from Base.utils.utils import setup_logger
from Base.utils.utils import set_seed, heatmap_gen_unicsl, save_results_unicsl, save_loss_frequencies
from Base.utils.load_data import load_datasets

from unicsl_single_vs_multi import main_unicsl

import copy, numpy as np
import time

filename = f"results/{SEED}_results_unicsl.xlsx"
set_seed()
log_file, tee = setup_logger(save_log=save_log)


client_schedule = [np.random.choice(num_clients, size=participants, replace=False) for _ in range(CR)]


def calculate_inverse(train_data_list):
    clients_Q_QXT = []
    for client in range(num_clients):
        X = train_data_list[client].iloc[:, :-1]
        I = np.eye(X.shape[1]).astype('int64')
        c = 10e-2
        Q = np.linalg.inv((c * I) + np.dot(X.T, X))
        Q_X_t = np.dot(Q, X.T)

        clients_Q_QXT.append((Q_X_t))
    
    return clients_Q_QXT


loss_functions = [
    # {'nonconvex exp loss3': [2, 3, 4]},
    # {'Least Squares': [2]},
    # {'Smooth Hinge': [10]},
    # {'Squared Hinge': [0]},
    # {'nonconvex exp loss3': [2, 3, 4]}
    {'Huber loss': [1]},
    ] 

for dataset in all_datasets:
    results_list = []
    train_data_list, global_test_data = load_datasets(dataset)
    features = train_data_list[0].shape[1] - 1
    global_parameters = np.random.rand(features)
    clients_Q_QXT = calculate_inverse(train_data_list)
    
    for loss_function in loss_functions:
        # results = (global_accuracy, total_iteration, client_iter_data, loss_dictionary, total_time, memory_used)
        results = main_unicsl(copy.deepcopy(global_parameters), copy.deepcopy(client_schedule), 
                                copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), copy.deepcopy(clients_Q_QXT), loss_function)
        print(f"unicsl_single with UniSVM and original data points Simulation complete on {dataset}")
        results_list.append(results)

    results = main_unicsl(copy.deepcopy(global_parameters), copy.deepcopy(client_schedule), 
                        copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), copy.deepcopy(clients_Q_QXT))
    results_list.append(results)

    save_results_unicsl(results_list, loss_functions, dataset, filename)
    # dynamic_loss_dict = results_list[-1][3]
    # save_loss_frequencies(dynamic_loss_dict, dataset, filename)

    # heatmap_gen_unicsl(results_list, loss_functions, dataset)

if log_file:
    log_file.close()
    tee.close()
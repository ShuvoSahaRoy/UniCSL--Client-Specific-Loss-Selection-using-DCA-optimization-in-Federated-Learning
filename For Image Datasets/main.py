import sys
# When called from run.py, a single aggregation name is passed as CLI argument.
# This ensures each algorithm runs in its own fresh process with clean memory.
selected_aggregation = sys.argv[1] if len(sys.argv) > 1 else None


import pandas as pd
import numpy as np
from config import *                       
from Base.utils.utils import (plot_and_write_function_frequencies, setup_logger, load_lr_rate, load_best_c,
                              generate_client_schedule, save_accuracy_metrics_to_excel)
from Base.utils.utils import set_seed, LinearSVM_OVO
from Base.utils.load_data import load_datasets


# Base line algorithms
from Base.baselines.fedavg import main_fedavg
from Base.baselines.fedprox import main_fedprox
from Base.baselines.scaffold import main_scaffold
from Base.baselines.fednova import main_fednova
from Base.baselines.feddyn import main_feddyn
from Base.baselines.fedopt import fedopt_main
from Base.baselines.pFedMe import main_pfedme
from Base.baselines.fedavg_multi_loss import main_fedavg_multi_loss
from Base.baselines.feddyn_multi_loss import main_feddyn_multi_loss

# Proposed algorithms
from Base.UniCSL.unicsl_static import main_unicsl as main_unicsl_static
# from Base.UniCSL.unicsl_dynamic import main_unicsl as main_unicsl_dynamic

from Base.utils.plot_result import plot_all
import copy
import time

set_seed(SEED)
log_file, tee = setup_logger(save_log=save_log)

results_list = []
client_schedule = generate_client_schedule(num_clients, participants, CR)

for dataset in all_datasets:
    print(f"\n{'='*20} Starting experiments on dataset: {dataset} {'='*20}\n")
    
    train_data_list, val_data_list, global_test_data = load_datasets(dataset)
    
    if dataset=='mnist' or dataset=='fashionmnist':
        num_classes, num_features = 10, 784
    elif dataset == 'cifar10':
        num_classes, num_features = 10, 2048
    elif dataset == 'femnist':
        num_classes, num_features = 62, 784
    global_parameters = LinearSVM_OVO(num_features, num_classes).to(device)

    for aggregation in aggregations:
        if selected_aggregation and aggregation != selected_aggregation:
            continue  # skip — this algorithm runs in its own process

        if aggregation != "unicsl_static":
            lr = load_lr_rate(dataset, aggregation, non_iid)
            print(f"lr rate for {dataset} in {aggregation} = {lr}")
        else:
            # for unicsl you can keep c as low as 0.01 for all datasets, it doesn't have a significant impact on performance, but you can also tune it for better results. We have provided the best c values we found for each dataset in 'c_rate.xlsx'.
            c, _ = load_best_c('c_rate.xlsx',SEED, dataset, non_iid, corrupt_data)

        if aggregation == 'fedavg':
            fedavg_acc = main_fedavg(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), client_schedule, lr)
            save_accuracy_metrics_to_excel(aggregation, fedavg_acc, dataset)
            print(f"FedAvg Simulation complete on {dataset} average_acc {np.mean(fedavg_acc[0])}")

        elif aggregation == 'fedprox':
            fedprox_acc = main_fedprox(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), client_schedule, lr)
            save_accuracy_metrics_to_excel(aggregation, fedprox_acc, dataset)
            print(f"Fedprox Simulation complete on {dataset} average_acc {np.mean(fedprox_acc[0])}")

        elif aggregation == 'scaffold':
            scaffold_acc = main_scaffold(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), client_schedule, lr)
            save_accuracy_metrics_to_excel(aggregation, scaffold_acc, dataset)
            print(f"Scaffold Simulation complete on {dataset} avg_acc {np.mean(scaffold_acc[0])}")
        
        elif aggregation == 'fednova':
            fednova_acc = main_fednova(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), client_schedule, lr)
            save_accuracy_metrics_to_excel(aggregation, fednova_acc, dataset)
            print(f"FedNova Simulation complete on {dataset} avg acc {np.mean(fednova_acc[0])}")    
        
        elif aggregation == 'feddyn':
            feddyn_acc = main_feddyn(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), client_schedule, lr)
            save_accuracy_metrics_to_excel(aggregation, feddyn_acc, dataset)
            print(f"FedDyn Simulation complete on {dataset}, avg acc {np.mean(feddyn_acc[0])}")  

        elif aggregation == 'fedopt':
            fedopt_acc = fedopt_main(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), client_schedule, lr)
            save_accuracy_metrics_to_excel(aggregation, fedopt_acc, dataset)
            print(f"FedOpt Simulation complete on {dataset} avg_acc {np.mean(fedopt_acc[0])}")   

        elif aggregation == 'pfedme':
            pfedme_acc = main_pfedme(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), client_schedule,
                        lr_outer = lr, personal_lr = 0.01, lam = 15, mu = mu,
                        R    = local_epoch, 
                        K    = 5,       # inner steps per outer iteration
                        beta        = 1.0)
            save_accuracy_metrics_to_excel(aggregation, pfedme_acc, dataset)
            print(f"pFedMe Simulation complete on {dataset}, avg acc {np.mean(pfedme_acc[0])}") 

        elif aggregation == 'unicsl_static':
            unicsl_static_acc, loss_dictionary = main_unicsl_static(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list),copy.deepcopy(val_data_list), copy.deepcopy(global_test_data),
                                             client_schedule, num_classes, c)
            save_accuracy_metrics_to_excel(aggregation, unicsl_static_acc, dataset)
            print(plot_and_write_function_frequencies(loss_dictionary, dataset))
            print(f"unicsl_static with UniSVM and original data points Simulation complete on {dataset}, avg acc {np.mean(unicsl_static_acc[0])}")

        elif aggregation == 'fedavg_multi_loss':
            fedavg_multi_loss_acc, loss_dictionary = main_fedavg_multi_loss(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list),copy.deepcopy(val_data_list), copy.deepcopy(global_test_data),
                                             client_schedule, lr)
            save_accuracy_metrics_to_excel(aggregation, fedavg_multi_loss_acc, dataset)
            print(plot_and_write_function_frequencies(loss_dictionary, dataset))
            print(f"FedAvg Multi Loss Simulation complete on {dataset}, avg acc {np.mean(fedavg_multi_loss_acc[0])}")

        elif aggregation == 'feddyn_multi_loss':
            feddyn_multi_loss_acc, loss_dictionary = main_feddyn_multi_loss(copy.deepcopy(global_parameters), copy.deepcopy(train_data_list),copy.deepcopy(val_data_list), copy.deepcopy(global_test_data),
                                             client_schedule, lr)
            save_accuracy_metrics_to_excel(aggregation, feddyn_multi_loss_acc, dataset)
            print(plot_and_write_function_frequencies(loss_dictionary, dataset))
            print(f"FedDyn Multi Loss Simulation complete on {dataset}, avg acc {np.mean(feddyn_multi_loss_acc[0])}")

if log_file:
    log_file.close()
    tee.close()
from config import *                       
from Base.utils.utils import setup_logger
from Base.utils.utils import set_seed
from Base.utils.load_data import load_datasets
from Base.utils.plot_result import plot_lr_tuning

# Base line algorithms
from Base.baselines.fedavg import main_fedavg
from Base.baselines.fedprox import main_fedprox
from Base.baselines.scaffold import main_scaffold
from Base.baselines.feddyn import main_feddyn
from Base.baselines.fedopt import main_fedopt
from Base.baselines.pfedme import main_pfedme   
from Base.UniCSL.unicsl_static import main_unicsl
# from Comparison_of_UniCSL.unicsl_single_vs_multi import main_unicsl as single_vs_multi_unicsl

# sq_hinge version of all implementation
from Base.baselines.sq_hinge.fedavg_sq_hinge import main_fedavg_sq_hinge
from Base.baselines.sq_hinge.fedprox_sq_hinge import main_fedprox_sq_hinge
from Base.baselines.sq_hinge.scaffold_sq_hinge import main_scaffold_sq_hinge
from Base.baselines.sq_hinge.feddyn_sq_hinge import main_feddyn_sq_hinge
from Base.baselines.sq_hinge.fedopt_sq_hinge import main_fedopt_sq_hinge
from Base.baselines.sq_hinge.pfedme_sq_hinge import main_pfedme_sq_hinge
import numpy as np  
import copy

set_seed()
log_file, tee = setup_logger(save_log=save_log,log_dir="results/logs/parameter_tunning")
lr_list = [0.9,0.5,0.25,0.1,0.05,0.01,0.001,0.0001]
client_schedule = [np.random.choice(num_clients, size=participants, replace=False) for _ in range(CR)]
results_list = []

for dataset in all_datasets:
    train_data_list, global_test_data = load_datasets(dataset)

    global_parameters = np.random.rand(train_data_list[0].shape[1])
    for aggregation in aggregations:
        accuracies = []
        for lr in lr_list:
            print(f"""Processing {dataset.split('.')[0]} dataset with {num_clients} clients {participants} participants non_iid={non_iid} 
                   alpha={alpha} CR={CR} local_epoch={local_epoch} lr {lr}""")

            # copy.deepcopy(global_parameters), client_schedule, copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr
            # fedavg with svm and original data point
            if aggregation == 'fedavg':
                FedAvg = main_fedavg(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
                print(f"FedAvg with logistic regression and original data points Simulation complete on {dataset}")
                accuracies.append(FedAvg[0])

            elif aggregation == 'fedavg_sq_hinge':
                FedAvg_sq_hinge = main_fedavg_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
                print(f"FedAvg with logistic regression and original data points Simulation complete on {dataset}")
                accuracies.append(FedAvg_sq_hinge[0])   

            elif aggregation == 'fedprox':
                FedProx = main_fedprox(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
                print(f"Fedprox with logistic regression and original data points Simulation complete on {dataset}")
                accuracies.append(FedProx[0])

            elif aggregation == 'fedprox_sq_hinge':
                FedProx_sq_hinge = main_fedprox_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
                print(f"Fedprox with logistic regression and original data points Simulation complete on {dataset}")
                accuracies.append(FedProx_sq_hinge[0])    

            elif aggregation == 'scaffold':
                SCAFFOLD = main_scaffold(   copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
                print(f"Scaffold with logistic regression and original data points Simulation complete on {dataset}")
                accuracies.append(SCAFFOLD[0])

            elif aggregation == 'scaffold_sq_hinge':
                SCAFFOLD_sq_hinge = main_scaffold_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
                print(f"Scaffold with logistic regression and original data points Simulation complete on {dataset}")
                accuracies.append(SCAFFOLD_sq_hinge[0])

            elif aggregation == "feddyn_sq_hinge":   
                feddyn_sq_hinge = main_feddyn_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
                print(f"feddyn with logistic regression and original data points Simulation complete on {dataset}")
                accuracies.append(feddyn_sq_hinge[0])

            elif aggregation == "fedopt_sq_hinge":
                fedopt_sq_hinge = main_fedopt_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
                print(f"fedopt with logistic regression and original data points Simulation complete on {dataset}")
                accuracies.append(fedopt_sq_hinge[0])

            elif aggregation == "pfedme_sq_hinge":
                pfedme_sq_hinge = main_pfedme_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr,
                                            lr_p = 0.01, lam = 15, mu = mu, R = local_epoch, K = 5, beta = 1.0)
                print(f"{aggregation} completed on {dataset} dataset. Avg Acc {np.mean(pfedme_sq_hinge[0])}, Time taken: {pfedme_sq_hinge[1]:.2f} seconds, Peak Memory Usage: {pfedme_sq_hinge[2]:.2f} MB")
                accuracies.append(pfedme_sq_hinge[0])   

            elif aggregation == "pfedme":
                pfedme = main_pfedme(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr,
                                            lr_p = 0.01, lam = 15, mu = mu, R = local_epoch, K = 5, beta = 1.0)
                print(f"{aggregation} completed on {dataset} dataset. Avg Acc {np.mean(pfedme[0])}, Time taken: {pfedme[1]:.2f} seconds, Peak Memory Usage: {pfedme[2]:.2f} MB")
                accuracies.append(pfedme[0])         

            # elif aggregation == 'unicsl':
            #     UniCSL = main_unicsl()
            #     print(f"unicsl with UniSVM and original data points Simulation complete on {dataset}")
            #     accuracies.append(UniCSL)
            
            # elif aggregation == 'unicsl_single':
            #     UniCSL_single, t_i_p_c, client_iter_data = single_vs_multi_unicsl(copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), lr, dataset, multi_loss=0)
            #     print(f"unicsl_v2 with UniSVM and original data points Simulation complete on {dataset}")
            #     accuracies.append(UniCSL_single)

            # elif aggregation == 'unicsl_multi':
            #     UniCSL_multi, t_i_p_c, client_iter_data = single_vs_multi_unicsl(copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), lr, dataset, multi_loss=1)
            #     print(f"unicsl with UniSVM and original data points Simulation complete on {dataset}")
            #     accuracies.append(UniCSL_multi)

        plot_lr_tuning(accuracies, lr_list, dataset, aggregation)

if log_file:
    log_file.close()
    tee.close()
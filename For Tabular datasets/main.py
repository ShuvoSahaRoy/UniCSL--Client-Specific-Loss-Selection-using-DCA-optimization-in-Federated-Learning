from config import local_epoch
import pandas as pd, numpy as np
from config import *                       
from Base.utils.utils import setup_logger
from Base.utils.utils import set_seed, save_experimental_results, save_loss_frequencies, load_lr_rate
from Base.utils.load_data import load_datasets

# Base line algorithms
from Base.baselines.fedavg import main_fedavg
from Base.baselines.fedprox import main_fedprox
from Base.baselines.scaffold import main_scaffold
from Base.baselines.feddyn import main_feddyn
from Base.baselines.fedopt import main_fedopt
from Base.baselines.pfedme import main_pfedme   

# sq_hinge version of all implementation
from Base.baselines.sq_hinge.fedavg_sq_hinge import main_fedavg_sq_hinge
from Base.baselines.sq_hinge.fedprox_sq_hinge import main_fedprox_sq_hinge
from Base.baselines.sq_hinge.scaffold_sq_hinge import main_scaffold_sq_hinge
from Base.baselines.sq_hinge.feddyn_sq_hinge import main_feddyn_sq_hinge
from Base.baselines.sq_hinge.fedopt_sq_hinge import main_fedopt_sq_hinge
from Base.baselines.sq_hinge.pfedme_sq_hinge import main_pfedme_sq_hinge

# Proposed algorithms
from Base.UniCSL.unicsl_static import main_unicsl as main_unicsl_static
from Base.UniCSL.unicsl_dynamic import main_unicsl as main_unicsl_dynamic


from Base.utils.plot_result import plot_all
# from Base.utils.utils import plot_similarity
import copy
import time

set_seed()
log_file, tee = setup_logger(save_log=save_log)
lr_df = pd.read_excel(r"lr_rate.xlsx")

client_schedule = [np.random.choice(num_clients, size=participants, replace=False) for _ in range(CR)]

for dataset in all_datasets:
    train_data_list, global_test_data = load_datasets(dataset)
    global_parameters = np.random.rand(train_data_list[0].shape[1])

    for aggregation in aggregations:

        # -----------------------------------------
        if aggregation != "unicsl_static":
            lr = load_lr_rate(dataset, aggregation, non_iid)
            print(f"lr rate for {dataset} in {aggregation} = {lr}")
            print(f"current lr {lr}")
        # lr = 0.1
        start_time = time.time()
        
        if aggregation == 'fedavg':
            fedavg = main_fedavg(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Time taken: {fedavg[1]:.2f} seconds, Peak Memory Usage: {fedavg[2]:.2f} MB")
            save_experimental_results(dataset, 'fedavg', fedavg, save)

        elif aggregation == 'fedavg_sq_hinge':
            fedavg_sq_hinge = main_fedavg_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Time taken: {fedavg_sq_hinge[1]:.2f} seconds, Peak Memory Usage: {fedavg_sq_hinge[2]:.2f} MB")
            save_experimental_results(dataset, 'fedavg_sq_hinge', fedavg_sq_hinge, save)


        elif aggregation == 'fedprox':
            fedprox = main_fedprox(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Time taken: {fedprox[1]:.2f} seconds, Peak Memory Usage: {fedprox[2]:.2f} MB")
            save_experimental_results(dataset, 'fedprox', fedprox, save)

        elif aggregation == 'fedprox_sq_hinge':
            fedprox_sq = main_fedprox_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Time taken: {fedprox_sq[1]:.2f} seconds, Peak Memory Usage: {fedprox_sq[2]:.2f} MB")
            save_experimental_results(dataset, 'fedprox_sq_hinge', fedprox_sq, save)    


        elif aggregation == 'scaffold':
            scaffold = main_scaffold(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Time taken: {scaffold[1]:.2f} seconds, Peak Memory Usage: {scaffold[2]:.2f} MB")
            save_experimental_results(dataset, 'scaffold', scaffold, save)

        elif aggregation == 'scaffold_sq_hinge':
            scaffold_sq_hinge = main_scaffold_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Time taken: {scaffold_sq_hinge[1]:.2f} seconds, Peak Memory Usage: {scaffold_sq_hinge[2]:.2f} MB")
            save_experimental_results(dataset, 'scaffold_sq_hinge', scaffold_sq_hinge, save)    
            
        elif aggregation == 'feddyn':
            feddyn = main_feddyn(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Avg Acc {np.mean(feddyn[0])}, Time taken: {feddyn[1]:.2f} seconds, Peak Memory Usage: {feddyn[2]:.2f} MB")
            save_experimental_results(dataset, 'feddyn', feddyn, save)

        elif aggregation == 'feddyn_sq_hinge':
            feddyn_sq_hinge = main_feddyn_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Avg Acc {np.mean(feddyn_sq_hinge[0])}, Time taken: {feddyn_sq_hinge[1]:.2f} seconds, Peak Memory Usage: {feddyn_sq_hinge[2]:.2f} MB")
            save_experimental_results(dataset, 'feddyn_sq_hinge', feddyn_sq_hinge, save)    


        elif aggregation == 'fedopt':
            fedopt = main_fedopt(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Avg Acc {np.mean(fedopt[0])}, Time taken: {fedopt[1]:.2f} seconds, Peak Memory Usage: {fedopt[2]:.2f} MB")
            save_experimental_results(dataset, 'fedopt', fedopt, save)

        elif aggregation == 'fedopt_sq_hinge':
            fedopt_sq_hinge = main_fedopt_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr)
            print(f"{aggregation} completed on {dataset} dataset. Avg Acc {np.mean(fedopt_sq_hinge[0])}, Time taken: {fedopt_sq_hinge[1]:.2f} seconds, Peak Memory Usage: {fedopt_sq_hinge[2]:.2f} MB")
            save_experimental_results(dataset, 'fedopt_sq_hinge', fedopt_sq_hinge, save)    


        elif aggregation == "pfedme":
            pfedme = main_pfedme(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr,
                                            lr_p = 0.01, lam = 15, mu = mu, R = local_epoch, K = 5, beta = 1.0)
            print(f"{aggregation} completed on {dataset} dataset. Avg Acc {np.mean(pfedme[0])}, Time taken: {pfedme[1]:.2f} seconds, Peak Memory Usage: {pfedme[2]:.2f} MB")
            save_experimental_results(dataset, 'pfedme', pfedme, save)


        elif aggregation == "pfedme_sq_hinge":
            pfedme_sq_hinge = main_pfedme_sq_hinge(copy.deepcopy(global_parameters), client_schedule,
                                             copy.deepcopy(train_data_list), copy.deepcopy(global_test_data[0]), lr,
                                            lr_p = 0.01, lam = 15, mu = mu, R = local_epoch, K = 5, beta = 1.0)
            print(f"{aggregation} completed on {dataset} dataset. Avg Acc {np.mean(pfedme_sq_hinge[0])}, Time taken: {pfedme_sq_hinge[1]:.2f} seconds, Peak Memory Usage: {pfedme_sq_hinge[2]:.2f} MB")
            save_experimental_results(dataset, 'pfedme_sq_hinge', pfedme_sq_hinge, save)    


        elif aggregation == 'unicsl_static':
            unicsl_static, loss_dictionary = main_unicsl_static(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data))
            print(f"{aggregation} completed on {dataset} dataset. Time taken: {unicsl_static[1]:.2f} seconds, Peak Memory Usage: {unicsl_static[2]:.2f} MB")
            save_experimental_results(dataset, 'unicsl_static', unicsl_static, save)
            # save_loss_frequencies(loss_dictionary, dataset)

        elif aggregation == 'unicsl_dynamic':
            unicsl_dynamic = main_unicsl_dynamic(copy.deepcopy(global_parameters), client_schedule,
                                 copy.deepcopy(train_data_list), copy.deepcopy(global_test_data))
            print(f"{aggregation} completed on {dataset} dataset. Time taken: {unicsl_dynamic[1]:.2f} seconds, Peak Memory Usage: {unicsl_dynamic[2]:.2f} MB")


if log_file:
    log_file.close()
    tee.close()
import pandas as pd
from config import *                       
from Base.utils.utils import setup_logger
from Base.utils.utils import set_seed, scenario_suffix, heatmap_gen
from Base.utils.load_data import load_datasets

from Base.UniCSL.unicsl_single_vs_multi import main_unicsl

from Base.utils.plot_result import plot_all
# from Base.utils.utils import plot_similarity
import copy
import time


set_seed()
log_file, tee = setup_logger(save_log=save_log)
lr_df = pd.read_excel(r"lr_rate.xlsx")
results_list = []

for dataset in all_datasets:
    train_data_list, global_test_data = load_datasets(dataset)
    
    for aggregation in aggregations:
        # -------- pick the learning rate ----------
        col_name = f"{aggregation.split('_')[0]} {scenario_suffix(noise, non_iid)}"
        try:
            lr = lr_df.loc[lr_df['dataset'] == dataset, col_name].values[0]
        except (KeyError, IndexError):
            print(f"[WARN] Column '{col_name}' or value for '{dataset}' not found.")
            lr = input(f"Please enter a learning rate for {dataset} with aggregation {aggregation}: ")
        # -----------------------------------------

        start_time = time.time()
        
        if aggregation == 'unicsl_single':
            UniCSL_v1, single_iter_per_cr, single_client_iter_data = main_unicsl(copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), lr, dataset,multi_loss = 0)
            print(f"unicsl_single with UniSVM and original data points Simulation complete on {dataset}")
        elif aggregation == 'unicsl_multi':
            UniCSL_v2, multi_iter_per_cr, multi_client_iter_data = main_unicsl(copy.deepcopy(train_data_list), copy.deepcopy(global_test_data), lr, dataset, multi_loss = 1)
            print(f"unicsl_multi with UniSVM and original data points Simulation complete on {dataset}")


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time for {aggregation} on {dataset}: {elapsed_time:.2f} seconds")
        # Store the result
        # results_list.append({"Dataset": dataset,"Algorithm": aggregation,"Non_IID": non_iid,"Execution Time (s)": elapsed_time})


    # results_df = pd.DataFrame(results_list)
    # print(results_df)

    # global_acc = [FedAvg, FedProx, SCAFFOLD, UniCSL]
    # algo_list = ['fedavg', 'fedprox', 'scaffold', 'unicsl']

    global_acc = [UniCSL_v1, UniCSL_v2]
    algo_list = ['UniCSL V1', 'UniCSL V2']

    plot_all(global_acc, algo_list, dataset)

    heatmap_gen(multi_client_iter_data,single_client_iter_data,dataset)

if log_file:
    log_file.close()
    tee.close()
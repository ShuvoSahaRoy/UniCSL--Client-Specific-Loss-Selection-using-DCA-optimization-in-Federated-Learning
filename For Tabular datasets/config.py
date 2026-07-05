SEED = 42

''''Federated Learning general settings'''
num_clients = 100
participants = 20
CR = 50
local_epoch = 10

'''Data distribution settings'''
non_iid = 1
alpha = 0.5
min_samples_per_client= 20

'''Noise and corruption settings'''
noise = 1
percentage_of_corrupt= 0.2
corrupted_clients = num_clients#//2

'''Aggregation specific settings'''
mu = 0.1  # Proximal term coefficient for FedProx
alpha_coef = 0.1  # Regularization coefficient for FedDyn


dataset_path = "./datasets/"

one_zero_dataset = ['A9a','phishing', '2dplanes', 'adult']    # "qsar_oral_toxicity",
continuous_datasets = ['magic_gamma_telescope','fried', 'Run_or_walk_information','skin_nonskin','hepmass', 'higgs','susy']
new_datasets = ['Default of Credit Card Clients']
# all_datasets = one_zero_dataset + continuous_datasets
# all_datasets = ['adult','2dplanes','fried','hepmass','magic_gamma_telescope','phishing']
# all_datasets = ['magic_gamma_telescope','A9a', 'Run_or_walk_information','hepmass','susy']
all_datasets = ['Run_or_walk_information']


"""List of aggregations to be used in the simulation
    main branchmark comparison ->> ['fedavg','fedprox', 'scaffold', 'unicsl_static']
    UniCSL loss selection variations ->> ['unicsl_static', 'unicsl_dynamic']
    UniCSL variations ->> ['unicsl_single', 'unicsl_multi']
"""
# aggregations = ['fedavg','fedprox', 'scaffold', 'feddyn', 'fedopt', 'pfedme', 'unicsl_static']
# aggregations = ['fedavg','fedprox', 'scaffold', 'unicsl_static']
# aggregations = ['unicsl_static', 'unicsl_dynamic']
# aggregations = ['unicsl_single', 'unicsl_multi']
# aggregations = ['unicsl_static']
# aggregations = ['fedavg','fedprox', 'scaffold', 'feddyn', 'fedopt']

# sq_hinge version of all implementation
aggregations = ['fedavg_sq_hinge','fedprox_sq_hinge','scaffold_sq_hinge','feddyn_sq_hinge','fedopt_sq_hinge', 'pfedme_sq_hinge', 'unicsl_static']
# aggregations = ['pfedme_sq_hinge']


save = 1
plot = 0
save_log = 0
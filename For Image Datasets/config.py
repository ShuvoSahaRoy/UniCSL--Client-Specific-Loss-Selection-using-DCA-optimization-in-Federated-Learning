import torch
SEED = 42

''''Federated Learning general settings'''
num_clients = 100
participants = 20
CR = 50
local_epoch = 10
batch_size = 1000000000

'''Data distribution settings'''
non_iid = 0
alpha = 0.5
min_samples_per_client= 20

corrupt_data = 0
corrupted_clients = num_clients
'''Validation settings: fraction of each client's training data to reserve for validation.'''
val_fraction = 0.1

# check and select the device: cuda, mps, cpu
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# device = 'cpu'


'''Aggregation specific settings'''
mu = 0.1  # Proximal term for FedProx, large mu means strong penalty/regularization/pulling towards global model.
          # in extrem heterogenity mu should be large for smooth convergernce but it will slow down the training. 
alpha_coef = 0.01 # Adaptive coefficient for FedDyn

dataset_path = "/Users/mlsilab/Documents/Shuvo/datasets/"
# all_datasets = ['mnist', 'fashionmnist', 'cifar10'] 
all_datasets = ['mnist'] 

# Validation data is mandatory for UniCSL
"""List of aggregations to be used in the simulation
    main branchmark comparison ->> ['fedavg','fedprox', 'scaffold', 'unicsl_static']
    UniCSL loss selection variations ->> ['unicsl_static', 'unicsl_dynamic']
    UniCSL variations ->> ['unicsl_single', 'unicsl_multi']
"""
# all baseline algo using square hinge loss
aggregations = ['fedavg','fedprox', 'scaffold', 'fednova', 'feddyn', 'fedopt', 'pfedme', 'unicsl_static']
# aggregations = ['fedavg','fedprox', 'scaffold', 'feddyn', 'fedopt', 'unicsl_static']
# aggregations = ['unicsl_static', 'unicsl_dynamic']
# aggregations = ['unicsl_single', 'unicsl_multi']
# aggregations = ['fedavg', 'fedavg_multi_loss', 'feddyn', 'feddyn_multi_loss']
# aggregations = ['pfedme']

#store results 
results_file = f'results/{SEED}_experimental_results.xlsx' 
save = 1
plot = 0
save_log = 0
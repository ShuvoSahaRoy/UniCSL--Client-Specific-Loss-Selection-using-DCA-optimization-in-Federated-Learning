import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy

from config import CR, num_clients, participants, local_epoch

# Neural network model with MLP structure
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

def get_model(input_dim):
    return MLP(input_dim)

def accuracy(model, test_data):
    model.eval()
    X = torch.tensor(test_data.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(test_data.iloc[:, -1].values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        y_pred = torch.sigmoid(logits)
        predictions = (y_pred > 0.5).float()
        correct = (predictions == y).sum().item()
        return 100 * correct / len(y)

def get_model_delta(model1, model2):
    delta = {}
    for k in model1.state_dict():
        delta[k] = model1.state_dict()[k] - model2.state_dict()[k]
    return delta

def apply_model_delta(model, delta, lr):
    new_state = model.state_dict()
    for k in delta:
        new_state[k] = new_state[k] - lr * delta[k]
    model.load_state_dict(new_state)
    return model

def client_update(global_model, data, lr):
    model = copy.deepcopy(global_model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for _ in range(local_epoch):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    delta = get_model_delta(global_model, model)
    return delta, len(data)

def SERVER(global_model, local_updates, lr, n):
    average_delta = {}
    for k in global_model.state_dict():
        average_delta[k] = sum(update[0][k] * (update[1] / n) for update in local_updates)
    global_model = apply_model_delta(global_model, average_delta, lr)
    return global_model

def main_fedavgNN(train_data_list, test_data, lr, dataset, base_model='NN', gb=False):

    if gb:
        train_data_list = process_gb_data_lr(train_data_list)

    features = train_data_list[0].iloc[:, :-1].shape[1]
    global_model = get_model(features)

    global_accuracy = []

    for round in range(CR):
        client_set = np.random.choice(num_clients, size=participants, replace=False)
        local_updates = []
        n = 0

        for client in client_set:
            client_data = train_data_list[client]
            local_update = client_update(global_model, client_data, lr)
            local_updates.append(local_update)
            n += len(client_data)

        global_model = SERVER(global_model, local_updates, lr, n)
        global_acc = accuracy(global_model, test_data)
        global_accuracy.append(global_acc)

        if (round + 1) % 20 == 0:
            print(f"{'-' * 20} communication round {round + 1} {'-' * 20}")
            print(f"{'-' * 20} {dataset} Global accuracy {global_acc:.4f} {'-' * 20}")

    return global_accuracy
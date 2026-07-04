import numpy as np
from config import local_epoch


def train(model, data, lr):

    w = model[:-1]
    b = model[-1]
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    y = np.where(y <= 0, -1, y)  # Ensure label format is correct
    lambda_param = 0.01

    for _ in range(local_epoch):
        for idx, x_i in enumerate(X):
            condition = y[idx] * (np.dot(x_i, w) - b) >= 1
            if condition:
                w -= lr * (2 * lambda_param * w)
            else:
                w -= lr * (2 * lambda_param * w - np.dot(x_i, y[idx]))
                b -= lr * y[idx]
    
    return np.concatenate((w, [b]))


def test(model_parameters, test_dataset):

    w = model_parameters[:-1]
    b = model_parameters[-1]
    X = test_dataset.iloc[:,:-1]
    y = test_dataset.iloc[:,-1]
    approx = np.dot(X, w) + b
    predictions = np.sign(approx)
    y_true = np.where(y <= 0, -1, y)  # Ensure label format is correct
    correct = np.sum(predictions == y_true)

    return (correct / len(test_dataset[1])) * 100
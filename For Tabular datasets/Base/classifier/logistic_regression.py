import numpy as np
from config import local_epoch

def train(model, data, lr):

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    w = model[:-1]
    b = model[-1]
    
    number_of_sample = X.shape[0]
    
    for i in range(local_epoch):
        linear = np.dot(X, w) + b
        y_pred = 1 / (1 + np.exp(-linear))
        
        # gradient of binary cross entropy
        dw = (1 / number_of_sample) * np.dot(X.T, (y_pred - y))
        db = (1 / number_of_sample) * np.sum(y_pred - y)
        
        w = w - lr * dw
        b = b - lr * db 
    
    return np.concatenate((w, [b]))


def test(model_parameters, test_dataset):

    w = model_parameters[:-1]
    b = model_parameters[-1]
    X = test_dataset.iloc[:, :-1]
    y = test_dataset.iloc[:, -1]

    linear = np.dot(X, w) + b
    y_predicted = 1 / (1 + np.exp(-linear))

    y_pred = [1 if i > 0.5 else 0 for i in y_predicted]
    test_accuracy = np.sum(y_pred == y)/len(y_pred) *100

    return test_accuracy
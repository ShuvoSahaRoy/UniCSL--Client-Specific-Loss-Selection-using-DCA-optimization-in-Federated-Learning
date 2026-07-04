# from time import perf_counter
import numpy as np 
from config import local_epoch, min_samples_per_client
from sklearn.metrics import f1_score, accuracy_score    
import itertools
from scipy.special import expit

class Loss_function():

    def __init__(self, str):
        self.result = None
        self.A = None
        self.str = str

    def Function(self, x, args):
        string = self.str
        if string == 'Least Squares':
            self.LeastSquares(x)
        elif string == 'Truncated LS':
            self.TruncatedLS(x, a=args[0])
        elif string == 'Truncated SH':
            self.TruncatedSH(x, a=args[0])
        elif string == 'Squared Hinge':
            self.SquaredHinge(x)
        elif string == 'Smooth Hinge':
            self.SmoothHinge(x, p=args[0])
        elif string == 'Smoothed Ramp1':
            self.SmoothedRamp1(x, a=args[0])
        elif string == 'Smoothed Ramp2':
            self.SmoothedRamp2(x, a=args[0], p=args[1])
        elif string == 'nonconvex exp loss1':
            self.nonconvex_exp_loss1(x, a=args[0], b=args[1], c=args[2])
        elif string == 'nonconvex exp loss2':
            self.nonconvex_exp_loss2(x, a=args[0], b=args[1], c=args[2])
        elif string == 'nonconvex exp loss3':
            self.nonconvex_exp_loss3(x, a=args[0], b=args[1], c=args[2])
        elif string == 'nonconvex log loss':
            self.nonconvex_log_loss(x, a=args[0], b=args[1], c=args[2])
        elif string == 'smooth eb insensitive':
            self.smooth_eb_insensitive(x, p=args[0], e=args[1])
        elif string == 'Huber loss':
            self.Huberloss(x, delta=args[0])
        elif string == 'smooth Absolute':
            self.smooth_Absolute(x, p=args[0])
        elif string == 'Smoothed TA1':
            self.SmoothedTA1(x, a=args[0])
        elif string == 'Smoothed TA2':
            self.SmoothedTA2(x, a=args[0], p=args[1])
        elif string == 'New Proposed eloss':
            self.New_Proposed_eloss(x, a=args[0], b=args[1], c=args[2])
        elif string == 'New Proposed gloss':
            self.New_Proposed_gloss(x, a=args[0], b=args[1], c=args[2])

    def exp1(self, derivative, x, a, b, c):
        x[x < 0] = 0
        if derivative:
            return -2 * x * np.exp((-1 / a) * x ** 2)
        else:
            return a * (1 - np.exp(-(x ** c) / b))

    def LeastSquares(self, x):
        self.A = 1
        self.result = 2 * x

    def TruncatedLS(self, x, a):
        self.A = 1
        x[np.abs(x) >= a ** 0.5] = 0
        self.result = 2 * x

    def TruncatedSH(self, x, a):
        self.A = 1
        x = np.maximum(x, 0)
        x[x > a ** 0.5] = 0
        self.result = 2 * x

    def SquaredHinge(self, x):
        self.A = 1
        x[x < 0] = 0
        self.result = 2 * x

    # def SmoothHinge(self, x, p):
    #     self.A = p / 8
    #     temp = np.exp(p * x)
    #     temp[temp > 1] = 1
    #     self.result = temp / (1 + np.exp(- p * np.abs(x)))
    def SmoothHinge(self, x, p):
        self.A = p / 8
        px = p * x
        px_capped = np.minimum(px, 0.0)
        temp = np.exp(px_capped)
        self.result = temp / (1 + np.exp(-p * np.abs(x)))

    def SmoothedRamp1(self, x, a):
        self.A = 2 / a
        x = np.where(x > a / 2, 4 / a * np.maximum(0, a - x), 4 / a * np.maximum(0, x))
        self.result = x

    # def SmoothedRamp2(self, x, a, p):
    #     self.A = p / 8
    #     t1 = np.exp(-p * (x - a))
    #     t2 = np.exp(-p * x)
    #     self.result = (t1 - t2) / ((1 + t1) * (1 + t2))
    def SmoothedRamp2(self, x, a, p):
        self.A = p / 8
        # expit(z) safely calculates 1 / (1 + exp(-z)) without ever overflowing
        self.result = expit(p * x) - expit(p * (x - a))

    def nonconvex_exp_loss1(self, x, a, b, c):
        self.A = 1
        t = np.maximum(0, x)
        self.result = (a * c) / b * t ** (c - 1) * np.exp(- (t ** c) / b)
    
    def nonconvex_exp_loss2(self, x, a, b, c):
        self.A = 1
        t = np.maximum(0, x)
        self.result = (a * c) / b * t ** (c - 1) * np.exp(- (t ** c) / b)
    
    def nonconvex_exp_loss3(self, x, a, b, c):
        self.A = 1
        t = np.maximum(0, x)
        self.result = (a * c) / b * t ** (c - 1) * np.exp(- (t ** c) / b)

    def nonconvex_log_loss(self, x, a, b, c):
        self.A = 1
        t = np.maximum(0, x)
        self.result = a * c * t ** (c - 1) / (b + t ** c)

    def smooth_eb_insensitive(self, x, p, e):
        self.A = p / 8
        t1 = np.exp(-p * (e - x))
        t2 = np.exp(-p * (e + x))
        self.result = 1 / (1 + t1) - 1 / (1 + t2)

    def Huberloss(self, x, delta):
        self.A = 1 / (2 * delta)
        x[np.abs(x) < delta] = 1 / (2 * delta) * x[np.abs(x) < delta]
        x[np.abs(x) >= delta] = x[np.abs(x) >= delta] / np.abs(x[np.abs(x) >= delta])
        self.result = x

    def smooth_Absolute(self, x, p):
        self.A = p / 8
        t = np.minimum(1, np.exp(p * x)) - np.minimum(1, np.exp(-p * x))
        self.result = t / (1 + np.exp(-p * np.abs(x)))

    def SmoothedTA1(self, x, a):
        self.A = 2 / a
        x[np.abs(x) <= a / 2] = 4 / a * x[np.abs(x) <= a / 2]
        x[np.abs(x) > a / 2] = 4 / a * np.maximum(a - np.abs(x[np.abs(x) > a / 2]), 0)
        self.result = x

    def SmoothedTA2(self, x, a, p):
        self.A = p / 8
        t1 = np.exp(-p * (x - a))
        t2 = np.exp(-p * x)
        self.result = (t1 - t2) / ((1 + t1) * (1 + t2))

    def New_Proposed_eloss(self, x, a, b, c):
        self.A = 1
        self.result = - (a * c / b) * x ** (c - 1) * np.exp(- (x ** c / b))

    def New_Proposed_gloss(self, x, a, b, c):
        self.A = 1
        self.result = (a * c * x ** (c - 1)) / (b + b / a * x ** c)



# def train(weights, data, client_Q_QXT, loss, num_classes=10):
#     """
#     weights: shape (features, num_pairs)
#     """
#     X_all = data.iloc[:, :-1].values
#     y_all = data.iloc[:, -1].values

#     pairs = list(itertools.combinations(range(num_classes), 2))
#     num_pairs = len(pairs)

#     loss_name = list(loss.keys())[0]
#     loss_params = list(loss.values())[0]
#     lf = Loss_function(loss_name)

#     new_weights = np.zeros_like(weights)

#     for k, (i, j) in enumerate(pairs):
#         # select samples for this pair
#         mask = np.logical_or(y_all == i, y_all == j)
#         X = X_all[mask]
#         y = y_all[mask]

#         if len(y) == 0:
#             new_weights[:, k] = weights[:, k]
#             continue

#         y_ij = np.where(y == i, 1, -1)
#         w = weights[:, k]

#         Q_X_T = client_Q_QXT[:, mask]

#         for _ in range(local_epoch):
#             temp = X @ w
#             x = 1 - y_ij * temp
#             lf.Function(x=x, args=loss_params)
#             gama = -0.5 / lf.A * y_ij * lf.result
#             w = Q_X_T @ (temp - gama)

#         new_weights[:, k] = w

#     return new_weights, len(X_all)


def train(weights, data, client_Q_QXT_dict, loss, num_classes=10):
    X_all = data.iloc[:, :-1].values
    y_all = data.iloc[:, -1].values

    pairs = list(itertools.combinations(range(num_classes), 2))
    
    loss_name = list(loss.keys())[0]
    loss_params = list(loss.values())[0]
    lf = Loss_function(loss_name)

    new_weights = np.zeros_like(weights)

    for k, (i, j) in enumerate(pairs):
        mask = np.logical_or(y_all == i, y_all == j)
        X = X_all[mask]
        y = y_all[mask]

        # Check if the client actually has data for this pair
        if len(y) == 0 or client_Q_QXT_dict[k] is None:
            new_weights[:, k] = weights[:, k] # Keep global weight static if no local data
            continue

        y_ij = np.where(y == i, 1, -1)
        w = weights[:, k]
        
        # Use the specific inverse for this pair
        Q_X_T = client_Q_QXT_dict[k] 

        for _ in range(local_epoch):
            temp = X @ w
            x = 1 - y_ij * temp
            lf.Function(x=x, args=loss_params)
            gama = -0.5 / lf.A * y_ij * lf.result
            w = Q_X_T @ (temp - gama)

        new_weights[:, k] = w

    return new_weights, len(X_all)


def test(model, test_data, num_classes=10):
    """
    model: shape (features, num_pairs)
    """
    X = test_data.iloc[:, :-1].values
    y_true = test_data.iloc[:, -1].values

    pairs = list(itertools.combinations(range(num_classes), 2))
    votes = np.zeros((X.shape[0], num_classes))

    scores = X @ model  # shape (N, num_pairs)

    for k, (i, j) in enumerate(pairs):
        preds = np.where(scores[:, k] > 0, i, j)
        for idx, p in enumerate(preds):
            votes[idx, p] += 1

    y_pred = np.argmax(votes, axis=1)

    accuracy = accuracy_score(y_true, y_pred) * 100
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    return accuracy, f1_macro

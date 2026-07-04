# from time import perf_counter
import numpy as np 
from config import local_epoch
import copy

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

    def SmoothHinge(self, x, p):
        self.A = p / 8
        temp = np.exp(p * x)
        temp[temp > 1] = 1
        self.result = temp / (1 + np.exp(- p * np.abs(x)))

    def SmoothedRamp1(self, x, a):
        self.A = 2 / a
        x = np.where(x > a / 2, 4 / a * np.maximum(0, a - x), 4 / a * np.maximum(0, x))
        self.result = x

    def SmoothedRamp2(self, x, a, p):
        self.A = p / 8
        t1 = np.exp(-p * (x - a))
        t2 = np.exp(-p * x)
        self.result = (t1 - t2) / ((1 + t1) * (1 + t2))

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



# train(weights.copy(), data, client_Q_QXT, best_loss, epoch)
def train(weights, data, client_Q_QXT, loss):
    c = 10e-8
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = np.where(y == 0, -1, y)
    
    loss_name = list(loss.keys())[0]
    loss_params = list(loss.values())[0]
    gama = np.zeros(X.shape[0])

    lf = Loss_function(loss_name)

    for run in range(local_epoch):
        temp = np.dot(X, weights)
        x = 1 - y * temp 
        lf.Function(x=x, args=loss_params)
        gama = -0.5 / lf.A * y * lf.result
        weights = np.dot(client_Q_QXT, (temp - gama))

    # weights = weights/ np.linalg.norm(weights)

    return weights, len(X)

def train_with_covergence(weights, data, client_Q_QXT, loss, c = 10e-8):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = np.where(y == 0, -1, y)
    
    loss_name = list(loss.keys())[0]
    loss_params = list(loss.values())[0]
    gama = np.zeros(X.shape[0])

    lf = Loss_function(loss_name)
    gama1 = copy.deepcopy(gama)
    iter = 0
    while True and iter<local_epoch:
        temp = np.dot(X, weights)
        x = 1 - y * temp 
        lf.Function(x=x, args=loss_params)
        gama = -0.5 / lf.A * y * lf.result
        weights = np.dot(client_Q_QXT, (temp - gama))

        iter += 1
        if np.linalg.norm(gama - gama1) < 1e-2:
            break
        gama1 = copy.deepcopy(gama)

    weights = weights/ np.linalg.norm(weights)
    return weights, len(X), iter


from sklearn.metrics import f1_score    
def test(model, test_data):
    X = test_data.iloc[:, :-1]
    y = test_data.iloc[:, -1]
    y = np.where(y == 0, -1, y)

    linear = np.dot(X, model)
    # margins = y - linear
    # total_loss = np.sum(np.abs(margins))/len(y)

    predictions = np.where(linear > 0, 1, -1)
    f1 = f1_score(y, predictions, pos_label=1, average='binary')

    correct_predictions = (predictions == y).astype(int)  # 1 if correct, 0 if wrong
    accuracy = np.mean(correct_predictions) * 100

    return accuracy, f1
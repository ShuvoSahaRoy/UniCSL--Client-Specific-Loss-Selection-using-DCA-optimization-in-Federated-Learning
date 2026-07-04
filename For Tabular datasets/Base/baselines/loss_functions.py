"""
Loss functions and their derivatives for logistic regression using NumPy.

Each loss function provides a `gradient(z, y)` method that computes dL/dz,
the gradient of the loss w.r.t. the linear combination z = Xw + b.

This gradient is used directly in SGD updates for logistic regression:
    dw = (1/n) * X.T @ gradient(z, y)
    db = (1/n) * sum(gradient(z, y))
    w  = w - lr * dw
    b  = b - lr * db

Adapted for federated learning with multi-loss selection based on F1 score.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────
def sigmoid(z):
    """Numerically stable sigmoid function."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )


# ──────────────────────────────────────────────────────────────────────
# Loss Function Classes  (each provides .gradient(z, y) -> dL/dz)
# ──────────────────────────────────────────────────────────────────────

class BinaryCrossEntropy:
    """Standard logistic regression loss (log loss).
    L  = -[ y*log(p) + (1-y)*log(1-p) ]   where p = sigmoid(z)
    dL/dz = p - y
    """
    name = 'Binary Cross-Entropy'

    @staticmethod
    def gradient(z, y):
        p = sigmoid(z)
        return p - y


class MeanSquaredError:
    """Mean Squared Error loss on the sigmoid output.
    L     = (p - y)^2
    dL/dp = 2(p - y)
    dL/dz = 2(p - y) * p * (1 - p)
    """
    name = 'Mean Squared Error'

    @staticmethod
    def gradient(z, y):
        p = sigmoid(z)
        return 2.0 * (p - y) * p * (1.0 - p)


class FocalLoss:
    """Focal Loss — down-weights easy examples, focuses on hard ones.
    L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where  p_t = p  if y=1, else  1-p.

    Parameters
    ----------
    gamma : float   focusing parameter (>=0, higher = more focus on hard)
    alpha : float   balancing factor for the positive class
    """
    name = 'Focal Loss'

    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def gradient(self, z, y):
        p = sigmoid(z)
        p = np.clip(p, 1e-7, 1.0 - 1e-7)

        # p_t: probability assigned to the *true* class
        p_t = np.where(y == 1, p, 1.0 - p)
        alpha_t = np.where(y == 1, self.alpha, 1.0 - self.alpha)

        # dL/dp_t  (derivative of focal loss w.r.t. p_t)
        log_pt = np.log(p_t)
        dL_dpt = -alpha_t * (
            self.gamma * ((1.0 - p_t) ** (self.gamma - 1)) * log_pt
            + ((1.0 - p_t) ** self.gamma) / p_t
        )

        # dp_t/dp:  +1 if y=1,  -1 if y=0
        dpt_dp = np.where(y == 1, 1.0, -1.0)

        # dp/dz = p * (1 - p)
        dp_dz = p * (1.0 - p)

        return dL_dpt * dpt_dp * dp_dz


class HingeLoss:
    """Hinge Loss (margin-based), standard SVM loss.
    Labels mapped to {-1, +1}: y_s = 2y - 1
    L     = max(0, 1 - y_s * z)
    dL/dz = -y_s   if  y_s * z < 1,  else 0
    """
    name = 'Hinge Loss'

    @staticmethod
    def gradient(z, y):
        y_s = 2.0 * y - 1.0
        margin = y_s * z
        return np.where(margin < 1.0, -y_s, 0.0)


class SquaredHingeLoss:
    """Squared Hinge Loss — differentiable everywhere.
    L     = max(0, 1 - y_s * z)^2
    dL/dz = -2 * y_s * max(0, 1 - y_s * z)
    """
    name = 'Squared Hinge Loss'

    @staticmethod
    def gradient(z, y):
        y_s = 2.0 * y - 1.0
        violation = np.maximum(0.0, 1.0 - y_s * z)
        return -2.0 * y_s * violation


class ExponentialLoss:
    """Exponential Loss (AdaBoost loss).
    L     = exp(-y_s * z)
    dL/dz = -y_s * exp(-y_s * z)
    Gradient is clipped to [-100, 100] for numerical stability.
    """
    name = 'Exponential Loss'

    @staticmethod
    def gradient(z, y):
        y_s = 2.0 * y - 1.0
        grad = -y_s * np.exp(-y_s * z)
        return np.clip(grad, -100.0, 100.0)


class SmoothHingeLoss:
    """Smooth approximation of Hinge Loss via log-sum-exp.
    L     = (1/beta) * log(1 + exp(-beta * (y_s * z - 1)))
    dL/dz = -y_s * sigma(-beta * (y_s * z - 1))
            where sigma is the sigmoid function.

    As beta -> inf, this approaches the standard Hinge loss.
    """
    name = 'Smooth Hinge Loss'

    def __init__(self, beta=5.0):
        self.beta = beta

    def gradient(self, z, y):
        y_s = 2.0 * y - 1.0
        margin = y_s * z - 1.0
        sig = sigmoid(-self.beta * margin)
        return -y_s * sig


class HuberLoss:
    """Huber Loss on the sigmoid output.
    Quadratic for small errors, linear for large — robust to outliers.

    dL/dz = dL/dp * p * (1-p)
    where dL/dp = (p-y)           if |p-y| <= delta
                  delta*sign(p-y)  otherwise
    """
    name = 'Huber Loss'

    def __init__(self, delta=0.5):
        self.delta = delta

    def gradient(self, z, y):
        p = sigmoid(z)
        diff = p - y
        abs_diff = np.abs(diff)

        dL_dp = np.where(abs_diff <= self.delta, diff, self.delta * np.sign(diff))
        return dL_dp * p * (1.0 - p)


class LogCoshLoss:
    """Log-Cosh Loss — smooth approximation of Huber loss.
    L     = log(cosh(p - y))
    dL/dp = tanh(p - y)
    dL/dz = tanh(p - y) * p * (1 - p)
    """
    name = 'Log-Cosh Loss'

    @staticmethod
    def gradient(z, y):
        p = sigmoid(z)
        return np.tanh(p - y) * p * (1.0 - p)


class TukeyBiweightLoss:
    """Tukey's Biweight Loss — highly robust, zero gradient for large errors.
    Operates on the sigmoid output residual r = (p - y).

    dL/dp = r * (1 - (r/c)^2)^2   if |r| <= c, else 0
    dL/dz = dL/dp * p * (1-p)
    """
    name = "Tukey's Biweight Loss"

    def __init__(self, c=0.5):
        self.c = c

    def gradient(self, z, y):
        p = sigmoid(z)
        r = p - y
        mask = np.abs(r) <= self.c
        ratio = (r / self.c) ** 2
        dL_dp = np.where(mask, r * (1.0 - ratio) ** 2, 0.0)
        return dL_dp * p * (1.0 - p)


class WeightedCrossEntropy:
    """Weighted Binary Cross-Entropy — assigns different cost to FP vs FN.
    L     = -[ w_pos * y * log(p)  +  w_neg * (1-y) * log(1-p) ]
    dL/dz = w_neg*(1-y)*p  -  w_pos*y*(1-p)
            (simplified using p*(1-p) cancellation with sigmoid derivative)
    """
    name = 'Weighted Cross-Entropy'

    def __init__(self, w_pos=2.0, w_neg=1.0):
        self.w_pos = w_pos
        self.w_neg = w_neg

    def gradient(self, z, y):
        p = sigmoid(z)
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        # Full derivation:
        # dL/dp = -w_pos * y / p  +  w_neg * (1-y) / (1-p)
        # dL/dz = dL/dp * p*(1-p)
        #       = -w_pos * y * (1-p)  +  w_neg * (1-y) * p
        return -self.w_pos * y * (1.0 - p) + self.w_neg * (1.0 - y) * p


# ──────────────────────────────────────────────────────────────────────
# Registry — list of dicts compatible with the FedAvg multi-loss selector
# ──────────────────────────────────────────────────────────────────────

LOSS_FUNCTIONS = [
    {'name': 'Binary Cross-Entropy',      'instance': BinaryCrossEntropy()},
    {'name': 'Mean Squared Error',         'instance': MeanSquaredError()},
    {'name': 'Focal Loss (γ=2, α=0.25)',   'instance': FocalLoss(gamma=2.0, alpha=0.25)},
    {'name': 'Focal Loss (γ=3, α=0.25)',   'instance': FocalLoss(gamma=3.0, alpha=0.25)},
    {'name': 'Hinge Loss',                 'instance': HingeLoss()},
    {'name': 'Squared Hinge Loss',         'instance': SquaredHingeLoss()},
    {'name': 'Exponential Loss',           'instance': ExponentialLoss()},
    {'name': 'Smooth Hinge (β=5)',         'instance': SmoothHingeLoss(beta=5.0)},
    {'name': 'Smooth Hinge (β=10)',        'instance': SmoothHingeLoss(beta=10.0)},
    {'name': 'Huber Loss (δ=0.5)',         'instance': HuberLoss(delta=0.5)},
    {'name': 'Log-Cosh Loss',              'instance': LogCoshLoss()},
    {"name": "Tukey's Biweight (c=0.5)",   'instance': TukeyBiweightLoss(c=0.5)},
    {'name': 'Weighted CE (2:1)',          'instance': WeightedCrossEntropy(w_pos=2.0, w_neg=1.0)},
]


def get_loss_function(name):
    """Retrieve a loss-function instance by its display name."""
    for entry in LOSS_FUNCTIONS:
        if entry['name'] == name:
            return entry['instance']
    available = [e['name'] for e in LOSS_FUNCTIONS]
    raise ValueError(f"Loss function '{name}' not found. Available: {available}")


def get_all_loss_functions():
    """Return every registered loss-function instance."""
    return [entry['instance'] for entry in LOSS_FUNCTIONS]


def get_all_loss_names():
    """Return a list of all registered loss-function names."""
    return [entry['name'] for entry in LOSS_FUNCTIONS]

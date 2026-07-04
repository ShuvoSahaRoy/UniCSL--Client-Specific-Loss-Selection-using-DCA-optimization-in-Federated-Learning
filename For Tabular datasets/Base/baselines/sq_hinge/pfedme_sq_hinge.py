from re import X
import numpy as np
import tracemalloc
import time
from config import *

# Reuse label conversion and test function from fedavg
from .fedavg_sq_hinge import _convert_labels, _sq_hinge_test


# ──────────────────────────────────────────────────────────────────────
#  pFedMe SERVER
#
#  wᵗ⁺¹ = (1-β)·wᵗ + β · (1/S) Σᵢ (nᵢ/n) · wᵗᵢ,R
#
#  Returns a NEW array — does not mutate global_model in-place.
#  (NumPy *= would make the intermediate state visible to the caller.)
#
#  Args:
#    global_model  : wᵗ  (1D numpy array)
#    local_models  : list of (wᵗᵢ,R, nᵢ)
#    n             : Σ nᵢ
#    beta          : β ≥ 1
# ──────────────────────────────────────────────────────────────────────
def pFedMe_SERVER(global_model, local_models, n, beta):
    new_global = (1.0 - beta) * global_model       # new array, no mutation
    for local_model, n_i in local_models:
        new_global += (beta * n_i / n) * local_model
    return new_global


# ──────────────────────────────────────────────────────────────────────
#  Pure task gradient — NO regularization baked in
#
#  ∂f/∂w = (1/n) Xᵀ · [-2y · max(0, 1-yz)]
#  ∂f/∂b = (1/n) Σ   [-2y · max(0, 1-yz)]
#
#  Regularization (µ) is handled exclusively by _pfedme_optimizer_step,
#  same as how pFedMeOptimizer works in the PyTorch version.
# ──────────────────────────────────────────────────────────────────────
def _sq_hinge_grad(theta, X, y):
    w = theta[:-1]
    b = theta[-1]
    n = X.shape[0]

    z         = np.dot(X, w) + b
    violation = np.maximum(0.0, 1.0 - y * z)
    grad_z    = -2.0 * y * violation

    dw = (1.0 / n) * np.dot(X.T, grad_z)
    db = (1.0 / n) * np.sum(grad_z)

    return np.concatenate((dw, [db]))


# ──────────────────────────────────────────────────────────────────────
#  pFedMeOptimizer step  (NumPy equivalent of pFedMeOptimizer.step())
#
#  θ ← θ - lr_p · ( ∇f(θ) + λ·(θ - w) + µ·θ )
#              └─ task ─┘  └─ proximal ─┘  └─ decay ─┘
#
#  µ applied exactly once here — not inside _sq_hinge_grad.
# ──────────────────────────────────────────────────────────────────────
def _pfedme_optimizer_step(theta, w, grad, lr_p, lam, mu):
    return theta - lr_p * (
        grad
        + lam * (theta - w)     # proximal: pulls θ toward current w
        + mu  *  theta          # weight decay — once only
    )


# ──────────────────────────────────────────────────────────────────────
#  CLIENT UPDATE
#
#  for r in range(R):
#      theta = w.copy()                    ← reset θ to current w
#      for k in range(K):
#          grad  = _sq_hinge_grad(theta, X, y)
#          theta = _pfedme_optimizer_step(theta, w, grad, lr_p, λ, µ)
#      w = w - λ·η·(w - θ̃ᵢ)              ← Moreau envelope gradient step
#
#  θ is reset at the start of every outer iteration r because w has
#  moved — each r solves a fresh subproblem anchored to the new w.
#
#  Args:
#    global_model : wᵗ  (not modified)
#    data         : client DataFrame  (features + last col = label)
#    lr_outer     : η    outer learning rate
#    lr_p         : η_p  personal learning rate
#    lam          : λ    Moreau regularization
#    mu           : µ    weight decay
#    R            : outer iterations per global round
#    K            : inner steps per outer iteration
# ──────────────────────────────────────────────────────────────────────
def client_update(global_model, data, lr_outer, lr_p, lam, mu, R, K):

    X         = data.iloc[:, :-1].values
    y         = _convert_labels(data.iloc[:, -1].values)   # {0,1} → {-1,+1}
    n_samples = X.shape[0]

    w = global_model.copy()

    theta = w.copy()      # initialize ONCE

    for r in range(R):

        for k in range(K):
            grad = _sq_hinge_grad(theta, X, y)
            theta = _pfedme_optimizer_step(theta, w, grad, lr_p, lam, mu)

        w = w - lam * lr_outer * (w - theta)

    return w, n_samples


# ──────────────────────────────────────────────────────────────────────
#  MAIN pFedMe
#
#  Same return structure as main_fedavg_sq_hinge():
#    (global_accuracy, total_time, peak_memory_MB)
#
#  Args:
#    global_parameters : w⁰  (1D numpy array, length = n_features + 1)
#    client_schedule   : list[list[int]], length CR
#    train_data_list   : list of per-client DataFrames
#    test_data         : centralized test DataFrame
#    lr_outer          : η       outer learning rate
#    lr_p              : η_p     personal learning rate
#    lam               : λ       Moreau regularization
#    mu                : µ       weight decay
#    R                 : outer iterations per global round
#    K                 : inner steps per outer iteration
#    beta              : β ≥ 1   server momentum
# ──────────────────────────────────────────────────────────────────────
def main_pfedme_sq_hinge(global_parameters, client_schedule, train_data_list,
                         test_data, lr_outer, lr_p, lam, mu, R, K, beta):

    global_accuracy = []

    start_time = time.time()
    tracemalloc.start()

    for round in range(CR):

        client_set   = client_schedule[round]
        local_models = []
        n = 0

        for client in client_set:
            w, n_i = client_update(
                global_parameters.copy(), train_data_list[client],
                lr_outer, lr_p, lam, mu, R, K
            )
            local_models.append((w, n_i))
            n += n_i

        # wᵗ⁺¹ = (1-β)·wᵗ + β·(1/S)Σ wᵗᵢ,R
        global_parameters = pFedMe_SERVER(global_parameters, local_models, n, beta)

        global_acc = _sq_hinge_test(global_parameters, test_data)
        global_accuracy.append(global_acc)
        print(f"[pFedMe-SqHinge] Round {round + 1}/{CR} | Acc: {global_acc:.2f}")

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time

    return (global_accuracy, total_time, (peak_memory / (1024 ** 2)))
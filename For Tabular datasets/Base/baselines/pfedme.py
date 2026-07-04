import numpy as np
import tracemalloc
import time
from config import *
from Base.classifier.logistic_regression import test


# ──────────────────────────────────────────────────────────────────────
#  pFedMe SERVER
#
#  wᵗ⁺¹ = (1-β)·wᵗ + β · (1/S) Σᵢ (nᵢ/n) · wᵗᵢ,R
#
#  Returns a NEW array — does not mutate global_model in-place.
#
#  Args:
#    global_model : wᵗ  (1D numpy array  [w; b])
#    local_models : list of (wᵗᵢ,R, nᵢ)
#    n            : Σ nᵢ
#    beta         : β ≥ 1  (β=1 → vanilla weighted average)
# ──────────────────────────────────────────────────────────────────────
def pFedMe_SERVER(global_model, local_models, n, beta):
    new_global = (1.0 - beta) * global_model       # new array, no mutation
    for local_model, n_i in local_models:
        new_global += (beta * n_i / n) * local_model
    return new_global


# ──────────────────────────────────────────────────────────────────────
#  Logistic Regression Gradient  (pure task gradient, no regularization)
#
#  Loss : L(θ) = -(1/n) Σ [ yᵢ·log(σ(z)) + (1-yᵢ)·log(1-σ(z)) ]
#         where z = Xw + b,  σ(z) = 1/(1+exp(-z))
#
#  ∂L/∂w = (1/n) Xᵀ·(σ(z) - y)
#  ∂L/∂b = (1/n) Σ (σ(z) - y)
#
#  No µ here — weight decay is handled exclusively in _pfedme_optimizer_step,
#  same separation as in the PyTorch pFedMeOptimizer.
#
#  Args:
#    theta : [w (d,); b (1,)]  concatenated → (d+1,)
#    X     : features  (n, d)
#    y     : labels in {0, 1}  (n,)
#
#  Returns:
#    grad  : (d+1,) array  [∂L/∂w; ∂L/∂b]
# ──────────────────────────────────────────────────────────────────────
def _logistic_grad(theta, X, y):
    w      = theta[:-1]
    b      = theta[-1]
    n      = X.shape[0]

    z      = np.dot(X, w) + b
    y_pred = 1.0 / (1.0 + np.exp(-z))      # sigmoid

    diff   = y_pred - y                     # (n,)

    dw     = (1.0 / n) * np.dot(X.T, diff)
    db     = (1.0 / n) * np.sum(diff)

    return np.concatenate((dw, [db]))


# ──────────────────────────────────────────────────────────────────────
#  pFedMeOptimizer step
#
#  θ ← θ - lr_p · ( ∇f(θ) + λ·(θ - w) + µ·θ )
#              └─ task ─┘  └─ proximal ─┘  └─ decay ─┘
#
#  µ applied exactly once here — not inside _logistic_grad.
#
#  Args:
#    theta : current personalized model  (d+1,)
#    w     : current local model anchor  (d+1,)
#    grad  : pure task gradient ∂f/∂θ    (d+1,)
#    lr_p  : η_p  personal learning rate
#    lam   : λ    Moreau regularization
#    mu    : µ    weight decay
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
#  Matches official UserpFedMe.train() — θ warm-started across R iters.
#
#    w     = global_model.copy()
#    theta = w.copy()              ← initialized once
#
#    for r in range(R):
#        for k in range(K):
#            grad  = ∂f/∂θ
#            theta = optimizer_step(theta, w, grad)
#        w = w - λ·η·(w - θ)      ← Moreau envelope gradient step
#
#  Args:
#    global_model : wᵗ  (not modified)
#    data         : client DataFrame  (features + last col = label {0,1})
#    lr_outer     : η    outer learning rate
#    lr_p         : η_p  personal learning rate
#    lam          : λ    Moreau regularization
#    mu           : µ    weight decay
#    R            : outer iterations per global round
#    K            : inner steps per outer iteration
#
#  Returns:
#    w            : wᵗᵢ,R  sent to server
#    n_samples    : local dataset size for weighted aggregation
# ──────────────────────────────────────────────────────────────────────
def client_update(global_model, data, lr_outer, lr_p, lam, mu, R, K):

    X         = data.iloc[:, :-1].values
    y         = data.iloc[:, -1].values        # labels stay in {0, 1}
    n_samples = X.shape[0]

    w     = global_model.copy()
    theta = w.copy()                           # warm-started across R iters

    for r in range(R):

        # ── K inner steps: θ → θ̃ᵢ(w) ──────────────────────────────────
        for k in range(K):
            grad  = _logistic_grad(theta, X, y)
            theta = _pfedme_optimizer_step(theta, w, grad, lr_p, lam, mu)

        # ── Outer step: w ← w - λ·η·(w - θ̃ᵢ) ───────────────────────────
        w = w - lam * lr_outer * (w - theta)

    return w, n_samples


# ──────────────────────────────────────────────────────────────────────
#  MAIN pFedMe  (logistic regression)
#
#  Same return structure as main_fedavg():
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
#    beta              : β ≥ 1   server momentum  (β=1 → vanilla avg)
# ──────────────────────────────────────────────────────────────────────
def main_pfedme(global_parameters, client_schedule, train_data_list,
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

        global_acc = test(global_parameters, test_data)
        global_accuracy.append(global_acc)
        print(f"[pFedMe] Round {round + 1}/{CR} | Acc: {global_acc:.2f}")

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time

    return (global_accuracy, total_time, (peak_memory / (1024 ** 2)))
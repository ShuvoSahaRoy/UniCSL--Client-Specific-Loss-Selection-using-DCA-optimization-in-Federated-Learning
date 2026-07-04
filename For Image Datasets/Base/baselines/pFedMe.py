import torch
import copy
import time
import psutil
import os
import gc
import numpy as np

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from config import device, CR
from .fedavg import test_ovo, squared_hinge_loss_ovo


# ─────────────────────────────────────────────────────────────────────────────
#  pFedMe SERVER
#
#  Differs from FedAvg SERVER in two ways:
#    1. Updates global_model IN-PLACE — no deepcopy, no new model allocation
#    2. Applies β momentum:
#         wᵗ⁺¹ = (1-β)·wᵗ + β · (1/S) Σᵢ (nᵢ/n)·wᵗᵢ,R
#
#  When β=1 → reduces to vanilla weighted average (same result as FedAvg SERVER)
#  When β>1 → overshoots the average (momentum), enabling quadratic speedup
#             as shown in Corollary 1 of the paper
#
#  Args:
#    global_model  : wᵗ — updated in-place to become wᵗ⁺¹
#    local_models  : list of (wᵗᵢ,R, nᵢ) from sampled clients
#    n             : total samples across sampled clients = Σ nᵢ
#    beta          : β ≥ 1 momentum parameter
# ─────────────────────────────────────────────────────────────────────────────
def pFedMe_SERVER(global_model, local_models, n, beta):
    with torch.no_grad():
        # Scale existing global model by (1-β)
        for g_p in global_model.parameters():
            g_p.data.mul_(1.0 - beta)
 
        # Accumulate β · (nᵢ/n) · wᵗᵢ,R for each client
        for local_model, n_i in local_models:
            weight = beta * n_i / n
            for g_p, l_p in zip(global_model.parameters(),
                                 local_model.parameters()):
                g_p.data.add_(l_p.data * weight)


# ─────────────────────────────────────────────────────────────────────────────
#  pFedMeOptimizer
#
#  Solves the inner problem for personalized model θ.
#  Update rule (one step):
#
#    θ ← θ - lr * ( ∇f(θ) + λ·(θ - w) + µ·θ )
#              └─ task ─┘  └─ proximal ─┘  └─ decay ─┘
#
#  Args:
#    params : θ parameters  (personalized model)
#    lr     : personal learning rate  η_p
#    lamda  : λ — Moreau regularization strength
#    mu     : µ — weight decay
# ─────────────────────────────────────────────────────────────────────────────
class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model_params, closure=None):
        """
        One gradient step on θ:
          θ ← θ - lr * ( ∇f(θ) + λ(θ - w) + µθ )

        local_model_params : list(local_model.parameters())  — live reference to w,
                             no cloning needed, not modified here.
        Updates θ (personalized_model.parameters()) in-place.
        Returns nothing — optimizer already owns the params.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p, w in zip(group['params'], local_model_params):
                if p.grad is None:
                    continue
                p.data = p.data - group['lr'] * (
                    p.grad.data
                    + group['lamda'] * (p.data - w.data)   # proximal term
                    + group['mu']    *  p.data              # weight decay
                )
        return loss


# ─────────────────────────────────────────────────────────────────────────────
#  CLIENT UPDATE
#
#  Structure (mirrors paper Algorithm 1 and official UserpFedMe.train()):
#
#    for r in range(R):                    ← R outer iterations (not local_epoch)
#        for k in range(K):                ← K inner steps to find θ̃ᵢ
#            forward → loss → backward
#            optimizer.step(w_params)      ← θ updated in-place
#        w ← w - λ·η·(w - θ)              ← Moreau envelope gradient step on w
#
#  Args:
#    global_model  : wᵗ from server  (not modified in-place)
#    data          : full client dataset  (DataLoader or DataFrame)
#    lr_outer      : η    outer learning rate for w update
#    personal_lr   : η_p  inner learning rate inside pFedMeOptimizer
#    lam           : λ    Moreau regularization
#    mu            : µ    weight decay
#    R             : outer iterations per global round
#    K             : inner steps per outer iteration to approximate θ̂ᵢ
#
#  Returns:
#    local_model   : wᵗᵢ,R  sent to server
#    n_samples     : dataset size for weighted aggregation
# ─────────────────────────────────────────────────────────────────────────────
def client_update(global_model, data, lr_outer, personal_lr, lam, mu, R, K):

    # ── Full-batch tensors (GD: entire dataset = one batch) ──────────────────
    if isinstance(data, DataLoader):
        X_list, y_list = [], []
        for xb, yb in data:
            X_list.append(xb)
            y_list.append(yb)
        X_full    = torch.cat(X_list).to(device)
        y_full    = torch.cat(y_list).to(device)
        n_samples = len(data.dataset)
    else:
        X_full    = torch.tensor(data.drop(columns=["y"]).values,
                                 dtype=torch.float32).to(device)
        y_full    = torch.tensor(data["y"].values,
                                 dtype=torch.long).to(device)
        n_samples = len(y_full)

    # ── w ← wᵗ  (local copy, updated R times) ────────────────────────────────
    local_model = copy.deepcopy(global_model)
    local_model.train()

    # ── θ ← w   (personalized model, updated K times per outer step) ─────────
    personalized_model = copy.deepcopy(local_model)
    personalized_model.train()

    optimizer = pFedMeOptimizer(
        personalized_model.parameters(),
        lr=personal_lr,
        lamda=lam,
        mu=mu
    )

    for r in range(R):

        # ── K inner steps: θ → θ̃ᵢ(w) ─────────────────────────────────────────
        # local_model.parameters() passed as live reference — no clone needed.
        # The optimizer reads w.data but never writes to it.
        for k in range(K):
            optimizer.zero_grad()

            logits = personalized_model(X_full)
            loss   = squared_hinge_loss_ovo(
                logits, y_full,
                personalized_model.pair_i,
                personalized_model.pair_j
            )
            loss.backward()

            # θ ← θ - lr_p * ( ∇f(θ) + λ(θ-w) + µθ )  — updates θ in-place
            optimizer.step(list(local_model.parameters()))

        # ── Outer step: w ← w - λ·η·(w - θ̃ᵢ) ───────────────────────────────
        with torch.no_grad():
            for w_p, theta_p in zip(local_model.parameters(),
                                     personalized_model.parameters()):
                w_p.data = w_p.data - lam * lr_outer * (w_p.data - theta_p.data)

    return local_model, n_samples


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN pFedMe
#
#  Tracks only pFedMe-GM accuracy (global model on centralized test set),
#  matching main_fedavg() return structure exactly.
#
#  Args:
#    global_parameters : initial global model w⁰
#    train_data_list   : list of per-client datasets  (len = N)
#    test_data         : centralized test set
#    client_schedule   : list[list[int]], length CR
#    lr_outer          : η       outer learning rate
#    personal_lr       : η_p     personal (inner) learning rate
#    lam               : λ       Moreau regularization
#    mu                : µ       weight decay
#    R                 : outer iterations per global round  (≠ local_epoch)
#    K                 : inner steps per outer iteration
#    beta              : β ≥ 1   server momentum  (β=1 → vanilla aggregation)
#
#  Returns:
#    (global_accuracy, total_time, memory_used)   ← same as main_fedavg()
# ─────────────────────────────────────────────────────────────────────────────
def main_pfedme(global_parameters, train_data_list, test_data,
                client_schedule, lr_outer, personal_lr, lam, mu, R, K, beta):

    global_accuracy = []

    # ── Profiling start ───────────────────────────────────────────────────────
    process = psutil.Process(os.getpid())
    gc.collect()
    mem_start  = process.memory_info().rss / (1024 ** 2)
    time_start = time.perf_counter()
    # ─────────────────────────────────────────────────────────────────────────

    for round in range(CR):

        client_set   = client_schedule[round]
        local_models = []    # (wᵗᵢ,R, n_i) → server aggregation
        n = 0

        for client in client_set:
            local_model, n_i = client_update(
                global_parameters, train_data_list[client],
                lr_outer, personal_lr, lam, mu, R, K
            )
            local_models.append((local_model, n_i))
            n += n_i

        # ── Server aggregation: wᵗ⁺¹ = (1-β)·wᵗ + β·(1/S)Σ wᵗᵢ,R ─────────
        # Updates global_parameters in-place — no deepcopy, no extra allocation
        pFedMe_SERVER(global_parameters, local_models, n, beta)
 
        # ── Centralized evaluation on global model (GM) ───────────────────────
        gm_acc = test_ovo(global_parameters, test_data)
        global_accuracy.append(gm_acc)
 
        print(f"Round {round} pFedMe | GM Acc: {gm_acc:.2f}")


    # ── Profiling end ─────────────────────────────────────────────────────────
    time_end    = time.perf_counter()
    mem_end     = process.memory_info().rss / (1024 ** 2)
    total_time  = time_end  - time_start
    memory_used = mem_end   - mem_start
    # ─────────────────────────────────────────────────────────────────────────

    print(f"pFedMe Total Time:   {total_time:.2f} seconds")
    print(f"pFedMe Memory Used:  {memory_used:.2f} MB")

    return (global_accuracy, total_time, memory_used)
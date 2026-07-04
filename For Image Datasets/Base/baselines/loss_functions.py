"""
PyTorch OvO (One-vs-One) loss functions for federated SVM training.

Each loss function follows the same interface as squared_hinge_loss_ovo:
    loss_fn(logits, targets, pair_i, pair_j, **extra_params) -> scalar loss

The numpy versions in unisvm.py compute **derivatives** (for manual gradient
computation).  Here we implement the **loss values** because PyTorch autograd
handles backpropagation automatically.

Conversion reference (derivative → loss, where x = 1 − y·logits):
  LeastSquares    :  2x             →  x²
  TruncatedLS     :  2x (|x|<√a)   →  min(x², a)
  TruncatedSH     :  2·relu(x<√a)  →  min(relu(x)², a)
  SquaredHinge    :  2·relu(x)      →  relu(x)²
  SmoothHinge     :  σ(px)          →  softplus(px) / p
  SmoothedRamp1   :  piecewise      →  piecewise quadratic
  SmoothedRamp2   :  σ(px)−σ(p(x−a))  →  [softplus(px)−softplus(p(x−a))] / p
  nonconvex_exp   :  (ac/b)t^(c−1)e^(−t^c/b)  →  a(1−e^(−t^c/b))
"""

import torch
import torch.nn.functional as F


# ====================================================================
# Helper: build OvO labels and mask from targets
# ====================================================================
def _ovo_margin(logits, targets, pair_i, pair_j):
    """
    Compute the OvO margin and mask.

    Returns:
        margin: (N, K) tensor,  margin = 1 - y * logits
        mask:   (N, K) tensor,  1.0 for active classifiers, 0.0 otherwise
    """
    t = targets.unsqueeze(1)  # (N, 1)
    y = (t == pair_i).float() - (t == pair_j).float()  # (N, K)  +1/-1/0
    mask = (y != 0).float()
    margin = 1 - y * logits
    return margin, mask


def _masked_mean(loss, mask):
    """Average the loss over active classifiers only."""
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# ====================================================================
# 1. Squared Hinge (reference – identical to fedavg.squared_hinge_loss_ovo)
# ====================================================================
def squared_hinge_loss_ovo(logits, targets, pair_i, pair_j):
    """
    f(x) = max(0, x)²      where x = 1 − y·logits
    f'(x) = 2·max(0, x)
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    loss = torch.clamp(margin, min=0.0) ** 2
    return _masked_mean(loss, mask)


# ====================================================================
# 2. Least Squares
# ====================================================================
def least_squares_loss_ovo(logits, targets, pair_i, pair_j):
    """
    f(x) = x²              where x = 1 − y·logits
    f'(x) = 2x
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    loss = margin ** 2
    return _masked_mean(loss, mask)


# ====================================================================
# 3. Truncated Least Squares
# ====================================================================
def truncated_ls_loss_ovo(logits, targets, pair_i, pair_j, a=1.0):
    """
    f(x) = min(x², a)      where x = 1 − y·logits
    f'(x) = 2x  if |x| < √a,  else 0

    Args:
        a: truncation threshold (loss is capped at a)
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    loss = torch.clamp(margin ** 2, max=a)
    return _masked_mean(loss, mask)


# ====================================================================
# 4. Truncated Squared Hinge
# ====================================================================
def truncated_sh_loss_ovo(logits, targets, pair_i, pair_j, a=1.0):
    """
    f(x) = min(max(0, x)², a)   where x = 1 − y·logits
    f'(x) = 2·max(0, x)  if x < √a,  else 0

    Args:
        a: truncation threshold
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    loss = torch.clamp(torch.clamp(margin, min=0.0) ** 2, max=a)
    return _masked_mean(loss, mask)


# ====================================================================
# 5. Smooth Hinge
# ====================================================================
def smooth_hinge_loss_ovo(logits, targets, pair_i, pair_j, p=1.0):
    """
    f(x) = softplus(p·x) / p  =  (1/p) ln(1 + e^{px})
    f'(x) = σ(p·x)            (sigmoid)

    Args:
        p: smoothness parameter (larger → sharper, approaches hinge)
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    loss = F.softplus(p * margin) / p
    # gamma = margin
    # loss = torch.clamp(gamma, min=0) + (1/p) * torch.log1p(torch.exp(-p * torch.abs(gamma)))
    return _masked_mean(loss, mask)


# ====================================================================
# 6. Smoothed Ramp 1
# ====================================================================
def smoothed_ramp1_loss_ovo(logits, targets, pair_i, pair_j, a=1.0):
    """
    Piecewise quadratic ramp loss:
        f(x) = 0                        if x ≤ 0
        f(x) = (2/a) x²                if 0 < x ≤ a/2
        f(x) = a − (2/a)(a − x)²       if a/2 < x < a
        f(x) = a                        if x ≥ a

    f'(x) = (4/a)·max(0, x)            if x ≤ a/2
    f'(x) = (4/a)·max(0, a − x)        if x > a/2

    Args:
        a: ramp width parameter
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    x = margin
    half_a = a / 2.0

    loss = torch.where(
        x <= 0,
        torch.zeros_like(x),
        torch.where(
            x <= half_a,
            (2.0 / a) * x ** 2,
            torch.where(
                x < a,
                a - (2.0 / a) * (a - x) ** 2,
                torch.full_like(x, a),
            ),
        ),
    )
    return _masked_mean(loss, mask)


# ====================================================================
# 7. Smoothed Ramp 2
# ====================================================================
def smoothed_ramp2_loss_ovo(logits, targets, pair_i, pair_j, a=1.0, p=1.0):
    """
    f(x) = [softplus(p·x) − softplus(p·(x − a))] / p
    f'(x) = σ(p·x) − σ(p·(x − a))

    Args:
        a: ramp width
        p: smoothness parameter
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    loss = (F.softplus(p * margin) - F.softplus(p * (margin - a))) / p
    return _masked_mean(loss, mask)


# ====================================================================
# 8. Non-convex Exponential Loss 1
# ====================================================================
def nonconvex_exp_loss1_ovo(logits, targets, pair_i, pair_j, a=1.0, b=1.0, c=2.0):
    """
    f(x) = a · (1 − exp(−max(0, x)^c / b))
    f'(x) = (a·c/b) · t^{c−1} · exp(−t^c / b),   t = max(0, x)

    Args:
        a: amplitude
        b: scale parameter
        c: exponent (c ≥ 1 recommended for smooth gradients)
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    t = torch.clamp(margin, min=0.0)
    loss = a * (1.0 - torch.exp(-(t ** c) / b))
    return _masked_mean(loss, mask)


# ====================================================================
# 9. Non-convex Exponential Loss 2
# ====================================================================
def nonconvex_exp_loss2_ovo(logits, targets, pair_i, pair_j, a=1.0, b=1.0, c=2.0):
    """
    f(x) = a · (1 − exp(−max(0, x)^c / b))
    f'(x) = (a·c/b) · t^{c−1} · exp(−t^c / b),   t = max(0, x)

    Note: Same formula as nonconvex_exp_loss1; they differ by parameter choice.

    Args:
        a: amplitude
        b: scale parameter
        c: exponent
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    t = torch.clamp(margin, min=0.0)
    loss = a * (1.0 - torch.exp(-(t ** c) / b))
    return _masked_mean(loss, mask)


# ====================================================================
# 10. Non-convex Exponential Loss 3
# ====================================================================
def nonconvex_exp_loss3_ovo(logits, targets, pair_i, pair_j, a=1.0, b=1.0, c=2.0):
    """
    f(x) = a · (1 − exp(−max(0, x)^c / b))
    f'(x) = (a·c/b) · t^{c−1} · exp(−t^c / b),   t = max(0, x)

    Note: Same formula as nonconvex_exp_loss1; they differ by parameter choice.

    Args:
        a: amplitude
        b: scale parameter
        c: exponent
    """
    margin, mask = _ovo_margin(logits, targets, pair_i, pair_j)
    t = torch.clamp(margin, min=0.0)
    loss = a * (1.0 - torch.exp(-(t ** c) / b))
    return _masked_mean(loss, mask)


# ====================================================================
# Convenience registry – look up a loss function by string name
# ====================================================================
LOSS_REGISTRY = {
    "Squared Hinge":        squared_hinge_loss_ovo,
    "Least Squares":        least_squares_loss_ovo,
    "Truncated LS":         truncated_ls_loss_ovo,
    "Truncated SH":         truncated_sh_loss_ovo,
    "Smooth Hinge":         smooth_hinge_loss_ovo,
    "Smoothed Ramp1":       smoothed_ramp1_loss_ovo,
    "Smoothed Ramp2":       smoothed_ramp2_loss_ovo,
    "nonconvex exp loss1":  nonconvex_exp_loss1_ovo,
    "nonconvex exp loss2":  nonconvex_exp_loss2_ovo,
    "nonconvex exp loss3":  nonconvex_exp_loss3_ovo,
}


def get_loss_fn(name):
    """
    Retrieve a loss function by its string name (matching unisvm.py naming).

    Usage:
        loss_fn = get_loss_fn("Smooth Hinge")
        loss = loss_fn(logits, targets, model.pair_i, model.pair_j, p=5.0)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name]

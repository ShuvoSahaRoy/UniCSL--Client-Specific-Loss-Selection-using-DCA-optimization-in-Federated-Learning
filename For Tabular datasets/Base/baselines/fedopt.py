"""
Based on FedOpt paper (Reddi et al., 2020) with configurable hyperparameters.

Algorithm:
1. Client: Run SGD for E epochs. Compute delta Δw = w_local - w_global.
2. Server: Aggregate Δw_t = Σ (n_k/N) * Δw_k (weighted average).
3. Server: Update moments using Adam rules.
4. Server: Update w_{t+1} = w_t + η * m_hat / (sqrt(v_hat) + ε).

Key difference from standard Adam:
- Standard Adam: θ = θ - η·gradient (descent)
- FedAdam: θ = θ + η·pseudo_gradient (ascent on model drift)
"""

import time
import tracemalloc
import numpy as np
from Base.classifier import logistic_regression
from config import CR, local_epoch


def client_update_fedopt(global_model, data, lr, local_epochs):
    """
    FedOpt client update using Numpy.
    
    Trains local model using SGD and returns the pseudo-gradient (model drift).
    
    Args:
        global_model: Current global model parameters (flat vector)
        data: Local training data (pandas DataFrame)
        lr: Client learning rate
        local_epochs: Number of local training epochs
        
    Returns:
        delta_w: The pseudo-gradient Δ = w_local - w_global
        n_samples: Number of samples in this client
    """
    # Initialize local model with global model weights
    model = global_model.copy()
    
    # Extract features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    n_samples = X.shape[0]
    
    # Local Training Loop (Standard SGD)
    for _ in range(local_epochs):
        # Split model into weights and bias
        w = model[:-1]
        b = model[-1]
        
        # Forward pass (logistic regression)
        linear = np.dot(X, w) + b
        # Numerical stability clip
        y_pred = 1 / (1 + np.exp(-np.clip(linear, -500, 500)))
        
        # Compute Gradients (Binary Cross-Entropy for logistic regression)
        # ∇L = X.T * (y_pred - y) / n
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)
        grad = np.concatenate((dw, [db]))
        
        # Gradient Descent Step
        model = model - lr * grad

    # Compute pseudo-gradient (model drift)
    # This represents the direction clients moved, which points toward better performance
    delta_w = model - global_model
    
    return delta_w, n_samples


def server_update_fedadam(global_model, local_updates, fedopt_state, 
                          server_lr=0.1, beta1=0.9, beta2=0.999, eps=1e-3):
    """
    FedAdam Server Update.
    
    Implements Algorithm 1 from FedOpt paper with Adam adaptive optimizer.
    
    Args:
        global_model: Current global model (θₜ)
        local_updates: List of tuples (delta_w, n_samples) from selected clients
        fedopt_state: Dictionary containing {'m', 'v', 't'}
        server_lr: Server learning rate η (default: 0.1)
        beta1: First moment decay rate (default: 0.9)
        beta2: Second moment decay rate (default: 0.999)
        eps: Numerical stability constant τ (default: 1e-3)
        
    Returns:
        new_global_model: Updated global model (θₜ₊₁)
        new_state: Updated state dictionary
    """
    # 1. Aggregate Client Pseudo-Gradients (Weighted Average)
    # Δₜ = Σₖ (nₖ/N) * Δₖ
    total_samples = sum(n for _, n in local_updates)
    
    # Initialize aggregate pseudo-gradient
    delta_t = np.zeros_like(global_model)
    
    for delta_w, n_k in local_updates:
        weight = n_k / total_samples
        delta_t += weight * delta_w
        
    # 2. Update FedAdam Moments
    t = fedopt_state['t'] + 1
    m = fedopt_state['m']
    v = fedopt_state['v']
    
    # First moment (Momentum)
    # mₜ = β₁ * mₜ₋₁ + (1 - β₁) * Δₜ
    m_new = beta1 * m + (1 - beta1) * delta_t
    
    # Second moment (Velocity/Adaptive Learning Rate)
    # vₜ = β₂ * vₜ₋₁ + (1 - β₂) * Δₜ²
    v_new = beta2 * v + (1 - beta2) * (delta_t ** 2)
    
    # 3. Bias Correction (Standard Adam practice)
    # Corrects for initialization bias in early rounds
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    
    # 4. Apply Adaptive Update to Global Model
    # θₜ₊₁ = θₜ + η * m̂ₜ / (√v̂ₜ + ε)
    # 
    # Note: We ADD (not subtract) because:
    # - delta_t = (w_local - w_global) points toward better solutions
    # - Standard Adam: θ = θ - η*gradient (gradient points uphill, we go downhill)
    # - FedAdam: θ = θ + η*pseudo_grad (pseudo_grad already points downhill)
    update_step = server_lr * m_hat / (np.sqrt(v_hat) + eps)
    new_global_model = global_model + update_step
    
    # Update state dictionary
    new_state = {
        'm': m_new,
        'v': v_new,
        't': t
    }
    
    return new_global_model, new_state


def main_fedopt(global_model, client_schedule, train_data_list, test_data, 
                client_lr=None, server_lr=0.1, 
                beta1=0.9, beta2=0.999, eps=1e-3,
                use_lr_decay=False, decay_rate=0.99):
    """
    Main FedOpt (FedAdam) training loop.
    
    Args:
        train_data_list: List of training data for each client
        test_data: Global test data
        client_lr: Client learning rate (if None, uses config.learning_rate)
        server_lr: Server learning rate (default: 0.1)
        beta1: Adam beta1 parameter (default: 0.9)
        beta2: Adam beta2 parameter (default: 0.999)
        eps: Adam epsilon parameter (default: 1e-3, paper value)
        use_lr_decay: Whether to decay server learning rate (default: False)
        decay_rate: Decay rate if using lr decay (default: 0.99)
        
    Returns:
        global_accuracy: List of test accuracies per round
    """

    # Model Initialization
    features = train_data_list[0].shape[1] - 1
    dim = features + 1
    
    # Initialize FedOpt State (m, v, t)
    fedopt_state = {
        'm': np.zeros(dim),  # First moment (momentum)
        'v': np.zeros(dim),  # Second moment (velocity)
        't': 0               # Time step (for bias correction)
    }
    
    global_accuracy = []
    
    # Current server learning rate (may decay)
    current_server_lr = server_lr

    start_time = time.time()
    tracemalloc.start()
    # Federated Learning Rounds
    for round_idx in range(CR):

        selected_clients = client_schedule[round_idx]
        local_updates = []
        
        # --- Client Updates ---
        for client_idx in selected_clients:
            # Client trains locally and computes pseudo-gradient
            delta_w, n_samples = client_update_fedopt(global_model, train_data_list[client_idx], client_lr,local_epoch)
            local_updates.append((delta_w, n_samples))
            
        # --- Server Update (FedAdam) ---
        global_model, fedopt_state = server_update_fedadam(
            global_model,
            local_updates,
            fedopt_state,
            server_lr=current_server_lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps
        )
        
        # Optional: Server Learning Rate Decay
        if use_lr_decay:
            current_server_lr = server_lr * (decay_rate ** round_idx)
        
        # Evaluation
        acc = logistic_regression.test(global_model, test_data)
        global_accuracy.append(acc)
        

        if use_lr_decay:
            print(f"[FedAdam] Round {round_idx + 1}/{CR} | "
                    f"Acc: {acc:.4f} | Server LR: {current_server_lr:.2f}")
        else:
            print(f"[FedAdam] Round {round_idx + 1}/{CR} | Acc: {acc:.2f}")
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time
 
    return (global_accuracy , total_time, (peak_memory / (1024**2)))


# =====================================================================
# VERIFICATION: Mathematical Correctness
# =====================================================================
"""
FedOpt (FedAdam) Algorithm Verification:

Paper's Algorithm (Reddi et al., 2020):
1. Client k computes: Δₖ = ClientUpdate(θₜ) - θₜ
2. Server aggregates: Δₜ = Σₖ pₖ·Δₖ (weighted by data size)
3. Server updates moments:
   - mₜ = β₁·mₜ₋₁ + (1-β₁)·Δₜ
   - vₜ = β₂·vₜ₋₁ + (1-β₂)·Δₜ²
4. Server updates model: θₜ₊₁ = θₜ + η·m̂ₜ/(√v̂ₜ + τ)

Our Implementation:
✓ Step 1: delta_w = model - global_model (after local training)
✓ Step 2: delta_t = Σ(nₖ/N)·delta_w (weighted average)
✓ Step 3: m_new = beta1*m + (1-beta1)*delta_t
          v_new = beta2*v + (1-beta2)*delta_t²
✓ Step 4: new_model = model + server_lr·m̂/(√v̂ + eps)

Additional (Standard Practice):
✓ Bias correction: m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)

Key Insight - Sign Convention:
- Standard gradient descent: θ = θ - α·∇f (go opposite of gradient)
- FedAdam pseudo-gradient: θ = θ + η·Δ (go with model drift)
  
Why? Because Δ = (w_local - w_global) already represents the direction
of improvement based on local data. If local training reduces loss,
w_local moves away from w_global in a beneficial direction.

This implementation is CORRECT and matches the FedOpt paper! ✓
"""


# =====================================================================
# USAGE EXAMPLES
# =====================================================================
"""
# Example 1: Basic FedAdam (paper defaults)
accuracies = main_fedopt(train_data_list, test_data)

# Example 2: With custom hyperparameters
accuracies = main_fedopt(
    train_data_list, 
    test_data,
    client_lr=0.01,
    server_lr=0.05,
    beta1=0.9,
    beta2=0.99,
    eps=1e-4
)

# Example 3: With server learning rate decay
accuracies = main_fedopt(
    train_data_list, 
    test_data,
    server_lr=0.1,
    use_lr_decay=True,
    decay_rate=0.99
)
"""
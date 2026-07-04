"""
1. Proper gradient formula: ∇L + α(θ - θ_prev + h_k)
2. Correct server update: θ^t = mean(selected θ) + mean(all h)
3. Proper h update: h_k^t = h_k^{t-1} + (θ_k^t - θ^{t-1})
"""

import time
import tracemalloc
import numpy as np
from Base.classifier import logistic_regression
from config import alpha_coef, num_clients, CR, local_epoch


def client_update_feddyn(global_model, data, h_k, theta_prev, alpha_k, lr):
    """
    FedDyn client update using numpy
    
    Implements gradient: ∇L_k(θ) + α_k*(θ - θ_prev + h_k)
    
    Args:
        global_model: Current global model (unused, kept for API consistency)
        data: Local training data (pandas DataFrame)
        h_k: Local history variable (model drift accumulator)
        theta_prev: Previous global model (what we regularize toward)
        alpha_k: Client-specific regularization coefficient
        lr: Learning rate
    
    Returns:
        theta_new: Updated local model
        new_h_k: Updated history variable
    """
    # Initialize local model from previous global model
    model = theta_prev.copy()
    
    # Extract features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    n = X.shape[0]
    
    # Training loop
    for _ in range(local_epoch):
        # Split model into weights and bias
        w = model[:-1]
        b = model[-1]
        
        # Forward pass (logistic regression)
        linear = np.dot(X, w) + b
        y_pred = 1 / (1 + np.exp(-np.clip(linear, -500, 500)))  # Numerical stability
        
        # Compute task loss gradient
        dw = (1/n) * np.dot(X.T, (y_pred - y))
        db = (1/n) * np.sum(y_pred - y)
        grad_task = np.concatenate((dw, [db]))
        
        # FedDyn regularization gradient
        # Gradient of: (α/2)||θ - θ_prev||² + α⟨h_k, θ⟩
        # = α(θ - θ_prev) + α*h_k = α(θ - θ_prev + h_k)
        grad_feddyn = alpha_k * (model - theta_prev + h_k)
        
        # Total gradient
        grad_total = grad_task + grad_feddyn
        
        # Gradient descent step
        model = model - lr * grad_total
    
    # Final local model
    theta_new = model
    
    # Update history variable: h_k^t = h_k^{t-1} + (θ_k^t - θ^{t-1})
    new_h_k = h_k + (theta_new - theta_prev)
    
    return theta_new, new_h_k


def server_update_feddyn(client_models, all_h_locals):
    """
    FedDyn server aggregation
    
    θ^t = (1/|S|) Σ_{k∈S} θ_k^t + (1/K) Σ_{k=1}^K h_k^t
    
    Args:
        client_models: Array of selected client models (shape: [num_selected, dim])
        all_h_locals: Array of ALL client history variables (shape: [num_clients, dim])
    
    Returns:
        theta_global: New global model
    """
    # Average of selected client models
    theta_bar = np.mean(client_models, axis=0)
    
    # Average of ALL client history variables
    h_mean = np.mean(all_h_locals, axis=0)
    
    # Global model update
    theta_global = theta_bar + h_mean
    
    return theta_global


def main_feddyn(global_model, client_schedule, train_data_list, test_data, lr=None):
    
    # Model dimensions
    features = train_data_list[0].shape[1] - 1
    dim = features + 1
    
    # Initialize history variables for all clients (h_k^0 = 0)
    client_h = np.zeros((num_clients, dim))
    
    # Store client models (for tracking)
    client_models = np.zeros((num_clients, dim))
    for k in range(num_clients):
        client_models[k] = global_model.copy()
    
    # Compute adaptive alpha based on data sizes (optional)
    sample_counts = np.array([len(data) for data in train_data_list])
    weight_list = sample_counts / np.sum(sample_counts) * num_clients
    use_adaptive_alpha = False  # Set to False for uniform alpha
    
    # Track accuracy
    global_accuracy = []
    
    start_time = time.time()
    tracemalloc.start()
    # Federated learning rounds
    for round_idx in range(CR):
        # Client selection
        selected_clients = client_schedule[round_idx]
        
        # Store current global model (this is theta_prev for all clients)
        theta_prev = global_model.copy()
        
        # Client updates
        selected_models = []
        for client_idx in selected_clients:
            # Adaptive or uniform alpha
            if use_adaptive_alpha:
                alpha_k = alpha_coef / weight_list[client_idx]
            else:
                alpha_k = alpha_coef
            
            # Local training
            theta_k, h_k = client_update_feddyn(
                global_model,              # Current global model (for API)
                train_data_list[client_idx],  # Local data
                client_h[client_idx],      # Current history
                theta_prev,                # Previous global model (regularization target)
                alpha_k,                   # Regularization coefficient
                lr                         # Learning rate
            )
            
            # Update stored values
            client_models[client_idx] = theta_k
            client_h[client_idx] = h_k
            selected_models.append(theta_k)
        
        # Convert to array
        selected_models = np.array(selected_models)
        
        # Server aggregation
        global_model = server_update_feddyn(selected_models, client_h)
        
        # Evaluation
        acc = logistic_regression.test(global_model, test_data)
        global_accuracy.append(acc)
        
        # Logging
        # if (round_idx + 1) % 10 == 0:
        print(f"[FedDyn] Round {round_idx + 1}/{CR} | Accuracy: {acc:.4f}")
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    total_time = end_time - start_time
 
    return (global_accuracy , total_time, (peak_memory / (1024**2)))


# =====================================================================
# VERIFICATION: Mathematical Correctness
# =====================================================================
"""
Gradient Computation Verification:

Target (from corrected PyTorch version):
    ∇L_k(θ) + α_k(θ - θ_prev) + α_k*h_k
    = ∇L_k(θ) + α_k*(θ - θ_prev + h_k)

Our Implementation:
    grad_task = ∇L_k(θ)                      # Task loss gradient
    grad_feddyn = α_k*(θ - θ_prev + h_k)     # FedDyn regularization
    grad_total = grad_task + grad_feddyn     # Combined gradient

✓ MATCHES EXACTLY!

Server Update Verification:

Target:
    θ^t = mean(selected θ_k) + mean(all h_k)

Our Implementation:
    theta_bar = np.mean(selected_models, axis=0)
    h_mean = np.mean(client_h, axis=0)
    theta_global = theta_bar + h_mean

✓ MATCHES EXACTLY!

History Update Verification:

Target:
    h_k^t = h_k^{t-1} + (θ_k^t - θ^{t-1})

Our Implementation:
    new_h_k = h_k + (theta_new - theta_prev)

✓ MATCHES EXACTLY!

This numpy implementation is mathematically equivalent to the corrected PyTorch version.
"""
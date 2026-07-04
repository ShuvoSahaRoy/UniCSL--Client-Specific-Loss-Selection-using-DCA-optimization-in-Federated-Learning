import numpy as np
import pandas as pd


def add_gaussian_noise_tabular(features_df, std_dev=0.3):
    """Add Gaussian noise to all features"""
    features = features_df.values.copy()  # Add .copy() - ALSO FIX THIS
    noise = np.random.normal(0, std_dev, features.shape)
    features = features + noise
    features = np.clip(features, -1, 1)
    return pd.DataFrame(features, columns=features_df.columns, index=features_df.index)


def flip_label_tabular(label, num_classes):
    """Flip a single label to a different class"""
    new_label = np.random.randint(0, num_classes)
    while new_label == label:
        new_label = np.random.randint(0, num_classes)
    return new_label


def introduce_outliers_tabular(features_df, fraction_of_features=0.3):
    """Introduce outliers to a fraction of features"""
    features = features_df.values.copy()  # Add .copy() - THIS IS THE FIX
    num_features = features.shape[1]
    
    # Compute Q1, Q3, and IQR for each feature
    Q1 = np.percentile(features, 25, axis=0)
    Q3 = np.percentile(features, 75, axis=0)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    
    # Select features to modify
    num_features_to_modify = int(fraction_of_features * num_features)
    feature_indices = np.random.choice(num_features, num_features_to_modify, replace=False)
    
    for col_idx in feature_indices:
        features[:, col_idx] = upper_bound[col_idx] if np.random.rand() > 0.5 else lower_bound[col_idx]
    
    return pd.DataFrame(features, columns=features_df.columns, index=features_df.index)


def corrupt_tabular_data(train_df, noise_type, corruption_rate=0.2, num_classes=None):
    """
    Corrupt tabular data with various noise types
    
    Args:
        train_df: DataFrame with features and 'y' column for labels
        noise_type: One of ['gaussian', 'label_flip', 'outliers']
        corruption_rate: Fraction of samples to corrupt (0.0 to 1.0)
        num_classes: Number of classes (required for label_flip)
    
    Returns:
        Corrupted DataFrame
    """
    train_df = train_df.copy()
    
    n_samples = len(train_df)
    n_corrupt = int(corruption_rate * n_samples)
    
    if n_corrupt == 0:
        return train_df
    
    # Select random indices to corrupt
    corrupt_indices = np.random.choice(n_samples, n_corrupt, replace=False)
    
    # Separate features and labels
    features = train_df.drop("y", axis=1).copy()  # Add .copy()
    labels = train_df["y"].values.copy()  # Add .copy() - THIS IS THE FIX
    
    if noise_type == "gaussian":
        # Add Gaussian noise to corrupted samples
        corrupted_features = features.loc[corrupt_indices].copy()
        corrupted_features = add_gaussian_noise_tabular(corrupted_features, std_dev=0.3)
        features.loc[corrupt_indices] = corrupted_features
    
    elif noise_type == "label_flip":
        # Flip labels for corrupted samples
        if num_classes is None:
            num_classes = len(np.unique(labels))
        
        for idx in corrupt_indices:
            labels[idx] = flip_label_tabular(labels[idx], num_classes)
    
    elif noise_type == "outliers":
        # Introduce outliers in corrupted samples
        corrupted_features = features.loc[corrupt_indices].copy()
        corrupted_features = introduce_outliers_tabular(corrupted_features, fraction_of_features=0.3)
        features.loc[corrupt_indices] = corrupted_features
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Use 'gaussian', 'label_flip', or 'outliers'")
    
    # Recombine features and labels
    train_df = features.copy()
    train_df["y"] = labels
    
    return train_df
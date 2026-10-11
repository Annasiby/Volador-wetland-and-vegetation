import numpy as np

def calculate_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Computes softmax probabilities from raw logits.
    
    Args:
        logits (np.ndarray): Input logits of shape (1, 4, H, W) or (4, H, W).
        
    Returns:
        np.ndarray: Softmax probabilities of shape (4, H, W).
    """
    if logits.ndim == 4:
        # logits shape is (1, 4, H, W)
        axis = 1
        max_logits = np.max(logits, axis=axis, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        sum_exp = np.sum(exp_logits, axis=axis, keepdims=True)
        probs = exp_logits / (sum_exp + 1e-12)
        # Squeeze batch dimension to return (4, H, W)
        return np.squeeze(probs, axis=0)
    elif logits.ndim == 3:
        # logits shape is (4, H, W)
        axis = 0
        max_logits = np.max(logits, axis=axis, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        sum_exp = np.sum(exp_logits, axis=axis, keepdims=True)
        return exp_logits / (sum_exp + 1e-12)
    else:
        raise ValueError(f"Unexpected logits dimensions: {logits.ndim}. Expected 3D or 4D array.")

def calculate_confidence(probabilities: np.ndarray) -> np.ndarray:
    """
    Computes per-pixel confidence as the maximum probability over classes.
    
    Args:
        probabilities (np.ndarray): Softmax probabilities of shape (4, H, W).
        
    Returns:
        np.ndarray: Confidence map of shape (H, W).
    """
    if probabilities.ndim != 3:
        raise ValueError(f"Expected probabilities shape (4, H, W), got {probabilities.shape}")
    return np.max(probabilities, axis=0)

"""
Entropy computation and temperature search.

The core math that maps logits onto the 1/x manifold. The key function is
`find_temperature_for_entropy()` — it binary-searches for a temperature τ
such that H(softmax(logits/τ)) ≈ target, which is how we enforce the
Horn constraint without modifying the model.
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor, base: float = 2.0) -> float:
    """
    Compute Shannon entropy of a logit vector (pre-softmax).

    Args:
        logits: Raw logit scores, shape [vocab_size] or [batch, vocab_size]
        base: Logarithm base (2.0 = bits, e = nats)

    Returns:
        Scalar entropy value in the specified base.
    """
    probs = F.softmax(logits.float(), dim=-1)
    probs = probs.clamp(min=1e-12)
    log_probs = torch.log(probs) / np.log(base)
    h = -(probs * log_probs).sum(dim=-1)
    return h.item()


def find_temperature_for_entropy(
    logits: torch.Tensor,
    target_entropy: float,
    tol: float = 0.01,
    max_iter: int = 50,
    t_low: float = 0.01,
    t_high: float = 5.0,
) -> float:
    """
    Binary search for temperature τ such that H(softmax(logits/τ)) ≈ target_entropy.

    This is the mechanism that "projects" the model's output distribution
    onto the Horn manifold. By controlling temperature, we control the
    effective radius (entropy) at each step.

    Args:
        logits: Raw logit scores, shape [vocab_size]
        target_entropy: Desired entropy value (in bits)
        tol: Acceptable error in entropy match
        max_iter: Maximum binary search iterations
        t_low: Lower temperature bound
        t_high: Upper temperature bound

    Returns:
        Temperature value that achieves (approximately) the target entropy.
    """
    for _ in range(max_iter):
        t_mid = (t_low + t_high) / 2.0
        h = compute_entropy(logits / t_mid)
        if abs(h - target_entropy) < tol:
            return t_mid
        if h > target_entropy:
            t_high = t_mid  # Lower temp → less entropy
        else:
            t_low = t_mid   # Higher temp → more entropy
    return (t_low + t_high) / 2.0

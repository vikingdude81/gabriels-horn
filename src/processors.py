"""
LogitsProcessor implementations for Gabriel's Horn Inference Experiment.

Core insight: Temperature scaling is a blunt instrument - it makes the entire
distribution peaky uniformly. When we crush temperature to force entropy down
the 1/x curve, we double down on whatever token was already most likely,
whether novel or repetitive.

The geometry of Gabriel's Horn suggests constraining the *shape* of search space,
not just its width. Selective suppression targets only tokens that would push
entropy beyond our target envelope - a surgical approach vs temperature's sledgehammer.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import LogitsProcessor

from .entropy import compute_entropy, find_temperature_for_entropy


class EntropyTracker:
    """Shared state for recording entropy at each generation step."""

    def __init__(self):
        self.raw_entropies: list[float] = []
        self.adjusted_entropies: list[float] = []
        self.target_entropies: list[float] = []
        self.temperatures: list[float] = []
        self.step: int = 0

    def reset(self):
        self.raw_entropies.clear()
        self.adjusted_entropies.clear()
        self.target_entropies.clear()
        self.temperatures.clear()
        self.step = 0


# ═══════════════════════════════════════════════════════════════════
# THE CLASSICAL HORN CONSTRAINT (temperature-based)
# ═══════════════════════════════════════════════════════════════════

class HornDecayProcessor(LogitsProcessor):
    """
    The Gabriel's Horn entropy constraint via temperature scaling.

    Forces sampling entropy to follow: H_target(n) = H_1 / f(n)

    WARNING: Temperature scales ALL logits uniformly - a sledgehammer approach.
    This may suppress novel tokens along with repetitive ones.

    Args:
        tracker: EntropyTracker for recording data
        decay_fn: Decay family — "inverse", "sqrt", or "log"
        floor_entropy: Minimum allowed entropy (prevents collapse)
        warmup_steps: Steps before constraint kicks in
        decay_multiplier: Scale factor for target entropy curve
    """

    def __init__(
        self,
        tracker: EntropyTracker,
        decay_fn: str = "sqrt",
        floor_entropy: float = 0.4,
        warmup_steps: int = 10,
        decay_multiplier: float = 1.0,
    ):
        self.tracker = tracker
        self.decay_fn = decay_fn
        self.floor_entropy = floor_entropy
        self.warmup_steps = warmup_steps
        self.decay_multiplier = decay_multiplier
        self.h1: float | None = None

    def _compute_decay(self, effective_n: int) -> float:
        if self.decay_fn == "inverse":
            return 1.0 / effective_n
        elif self.decay_fn == "sqrt":
            return 1.0 / np.sqrt(effective_n)
        elif self.decay_fn == "log":
            return 1.0 / np.log(effective_n + 1)
        return 1.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.tracker.step += 1
        n = self.tracker.step
        logits = scores[0] if scores.dim() > 1 else scores

        h_raw = compute_entropy(logits)
        self.tracker.raw_entropies.append(h_raw)

        if self.h1 is None:
            self.h1 = h_raw

        # Warmup phase: pass through unmodified
        if n <= self.warmup_steps:
            self.tracker.target_entropies.append(h_raw)
            self.tracker.adjusted_entropies.append(h_raw)
            self.tracker.temperatures.append(1.0)
            return scores

        # Compute target entropy from decay schedule
        effective_n = n - self.warmup_steps
        decay = self._compute_decay(effective_n)
        h_target = max(self.h1 * decay * self.decay_multiplier, self.floor_entropy)
        self.tracker.target_entropies.append(h_target)

        if h_raw <= h_target:
            self.tracker.adjusted_entropies.append(h_raw)
            self.tracker.temperatures.append(1.0)
            return scores

        # Temperature scaling to achieve target entropy
        tau = find_temperature_for_entropy(logits, h_target)
        adjusted_logits = logits / tau
        h_adj = compute_entropy(adjusted_logits)

        self.tracker.adjusted_entropies.append(h_adj)
        self.tracker.temperatures.append(tau)

        if scores.dim() > 1:
            return scores / tau
        return adjusted_logits.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════
# SELECTIVE HORN CONSTRAINT (NEW - surgical suppression)
# ═══════════════════════════════════════════════════════════════════

class HornSelectiveProcessor(LogitsProcessor):
    """
    Selective Horn constraint via token suppression, not temperature scaling.

    KEY INSIGHT: Instead of scaling ALL logits uniformly (which suppresses both
    novel and repetitive tokens), compute which tokens would push entropy beyond
    the target envelope and only suppress those. This matches the geometric intent:
    constraining trajectory shape rather than distribution width.

    Algorithm:
        1. Compute current entropy H_raw
        2. If H_raw > H_target, find and suppress high-probability tokens that
           would keep entropy too low (the "repetition risk" tokens)
        3. Apply minimal suppression needed to reach target envelope
    
    Args:
        tracker: EntropyTracker for recording data
        decay_fn: Decay family — "inverse", "sqrt", or "log"
        floor_entropy: Minimum allowed entropy
        warmup_steps: Steps before constraint kicks in
        decay_multiplier: Scale factor for target entropy
        suppression_strength: How aggressively to suppress (0.1-2.0)
    """

    def __init__(
        self,
        tracker: EntropyTracker,
        decay_fn: str = "sqrt",
        floor_entropy: float = 0.4,
        warmup_steps: int = 10,
        decay_multiplier: float = 1.0,
        suppression_strength: float = 0.5,
    ):
        self.tracker = tracker
        self.decay_fn = decay_fn
        self.floor_entropy = floor_entropy
        self.warmup_steps = warmup_steps
        self.decay_multiplier = decay_multiplier
        self.suppression_strength = max(0.1, min(suppression_strength, 2.0))
        self.h1: float | None = None

    def _compute_decay(self, effective_n: int) -> float:
        if self.decay_fn == "inverse":
            return 1.0 / effective_n
        elif self.decay_fn == "sqrt":
            return 1.0 / np.sqrt(effective_n)
        elif self.decay_fn == "log":
            return 1.0 / np.log(effective_n + 1)
        return 1.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.tracker.step += 1
        n = self.tracker.step
        logits = scores[0] if scores.dim() > 1 else scores.clone()

        h_raw = compute_entropy(logits)
        self.tracker.raw_entropies.append(h_raw)

        if self.h1 is None:
            self.h1 = h_raw

        # Warmup phase: pass through unmodified
        if n <= self.warmup_steps:
            self.tracker.target_entropies.append(h_raw)
            self.tracker.adjusted_entropies.append(h_raw)
            self.tracker.temperatures.append(1.0)
            return scores

        # Compute target entropy from decay schedule
        effective_n = n - self.warmup_steps
        decay = self._compute_decay(effective_n)
        h_target = max(self.h1 * decay * self.decay_multiplier, self.floor_entropy)
        self.tracker.target_entropies.append(h_target)

        # If already at or below target, no suppression needed
        if h_raw <= h_target:
            self.tracker.adjusted_entropies.append(h_raw)
            self.tracker.temperatures.append(1.0)
            return scores

        # Find tokens that would keep entropy too low (high-probability ones)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(5, len(probs)))
        
        # Suppress high-probability tokens to increase entropy toward target
        suppression = self.suppression_strength * (h_raw - h_target) / max(h_raw, 0.01)
        for idx in top_indices:
            logits[idx] -= suppression

        # Re-normalize and compute new entropy
        adjusted_logits = torch.softmax(logits, dim=-1)
        h_adj = -torch.sum(adjusted_logits * torch.log(adjusted_logits + 1e-10))

        self.tracker.adjusted_entropies.append(h_adj.item())
        self.tracker.temperatures.append(1.0)  # No temperature change in selective mode

        if scores.dim() > 1:
            return logits.unsqueeze(0)
        return logits


# ═══════════════════════════════════════════════════════════════════
# COMBO PROCESSOR (Horn + Repetition Penalty)
# ═══════════════════════════════════════════════════════════════════

class HornRepPenaltyProcessor(LogitsProcessor):
    """
    Hybrid: Horn entropy constraint combined with repetition penalty.

    This applies BOTH mechanisms simultaneously:
        1. Repetition penalty: suppresses tokens that already appeared in generation
        2. Horn decay: adjusts temperature to follow 1/√x entropy schedule
    
    The hypothesis: these are complementary, not redundant. Rep penalty handles
    surface-level repetition while Horn constrains the search space geometry.

    IMPORTANT: This processor MUST be used WITHOUT setting 'repetition_penalty' 
    in model.generate(), as it applies rep penalty internally.

    Args:
        tracker: EntropyTracker for recording data
        rep_penalty: Penalty factor >1 suppresses repeated tokens (e.g., 1.2)
        decay_fn: Decay family — "inverse", "sqrt", or "log"
        floor_entropy: Minimum allowed entropy
        warmup_steps: Steps before constraint kicks in
    """

    def __init__(
        self,
        tracker: EntropyTracker,
        rep_penalty: float = 1.2,
        decay_fn: str = "sqrt",
        floor_entropy: float = 0.4,
        warmup_steps: int = 10,
    ):
        self.tracker = tracker
        self.rep_penalty = rep_penalty
        self.decay_fn = decay_fn
        self.floor_entropy = floor_entropy
        self.warmup_steps = warmup_steps
        self.h1: float | None = None

    def _compute_decay(self, effective_n: int) -> float:
        if self.decay_fn == "inverse":
            return 1.0 / effective_n
        elif self.decay_fn == "sqrt":
            return 1.0 / np.sqrt(effective_n)
        elif self.decay_fn == "log":
            return 1.0 / np.log(effective_n + 1)
        return 1.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """Apply rep penalty first (via _apply_rep_penalty), then Horn constraint.
        
        CRITICAL: Must return tensors with EXACT same shape as input scores to avoid
        dimension errors in transformers generation loop.
        """
        # Track entropy on original scores during warmup (before any modification)
        self.tracker.step += 1
        n = self.tracker.step
        
        h_raw = compute_entropy(scores[0] if scores.dim() > 1 else scores)
        self.tracker.raw_entropies.append(h_raw)

        if self.h1 is None:
            self.h1 = h_raw

        # Warmup phase: apply rep penalty but no temperature scaling
        if n <= self.warmup_steps:
            self.tracker.target_entropies.append(float("nan"))
            self.tracker.adjusted_entropies.append(h_raw)
            self.tracker.temperatures.append(1.0)
            # Apply rep penalty to scores and return with SAME shape as input
            penalized = self._apply_rep_penalty_to_shape(scores, input_ids)
            return penalized

        # Compute target entropy from decay schedule (on raw entropy of current step)
        effective_n = n - self.warmup_steps
        decay = self._compute_decay(effective_n)
        h_target = max(self.h1 * decay, self.floor_entropy)
        self.tracker.target_entropies.append(h_target)

        if h_raw <= h_target:
            # Already below target - apply rep penalty but no scaling
            self.tracker.adjusted_entropies.append(h_raw)
            self.tracker.temperatures.append(1.0)
            penalized = self._apply_rep_penalty_to_shape(scores, input_ids)
            return penalized

        # Temperature scaling: compute tau on base logits, then scale scores directly
        base_logits = scores[0] if scores.dim() > 1 else scores
        tau = find_temperature_for_entropy(base_logits, h_target)
        
        self.tracker.adjusted_entropies.append(h_raw / tau)
        self.tracker.temperatures.append(tau)

        # Scale scores directly (preserves shape), then apply rep penalty to scaled logits
        scaled_scores = scores / tau if scores.dim() > 1 else scores.unsqueeze(0) / tau
        
        # Apply rep penalty to the already-scaled logits
        penalized_scaled = self._apply_rep_penalty_to_shape(scaled_scores, input_ids)
        
        return penalized_scaled

    def _apply_rep_penalty_to_shape(self, scores: torch.FloatTensor, input_ids: torch.LongTensor):
        """Apply repetition penalty while preserving tensor shape."""
        if scores.dim() > 1:
            # Batched case: apply to first element and broadcast back
            base = scores[0].clone()
            prev_tokens = set(input_ids[0].tolist())
            for token_id in prev_tokens:
                if token_id not in [0, 2]:
                    base[token_id] /= self.rep_penalty
            return base.unsqueeze(0) if len(scores.shape) == 3 else scores.clone()
        else:
            # Unbatched case
            base = scores.clone()
            prev_tokens = set(input_ids[0].tolist())
            for token_id in prev_tokens:
                if token_id not in [0, 2]:
                    base[token_id] /= self.rep_penalty
            return base.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════

class ConstantTemperatureProcessor(LogitsProcessor):
    """Baseline: constant temperature at every step."""

    def __init__(self, tracker: EntropyTracker, temperature: float = 0.7):
        self.tracker = tracker
        self.temperature = temperature

    def __call__(self, input_ids, scores):
        self.tracker.step += 1
        logits = scores[0] if scores.dim() > 1 else scores

        h_raw = compute_entropy(logits)
        self.tracker.raw_entropies.append(h_raw)

        adjusted = logits / self.temperature
        h_adj = compute_entropy(adjusted)

        self.tracker.target_entropies.append(float("nan"))
        self.tracker.adjusted_entropies.append(h_adj)
        self.tracker.temperatures.append(self.temperature)

        if scores.dim() > 1:
            return scores / self.temperature
        return adjusted.unsqueeze(0)


class LinearDecayProcessor(LogitsProcessor):
    """Baseline: linear temperature decay with stability safeguards."""

    def __init__(
        self,
        tracker: EntropyTracker,
        t_start: float = 1.0,
        t_end: float = 0.3,
        max_steps: int = 2000,
        min_temperature: float = 0.25,
    ):
        self.tracker = tracker
        self.t_start = t_start
        self.t_end = max(t_end, min_temperature)
        self.max_steps = max_steps
        self.min_temperature = min_temperature

    def __call__(self, input_ids, scores):
        self.tracker.step += 1
        n = self.tracker.step
        logits = scores[0] if scores.dim() > 1 else scores

        h_raw = compute_entropy(logits)
        self.tracker.raw_entropies.append(h_raw)

        progress = min(n / self.max_steps, 1.0)
        tau = max(self.t_start + (self.t_end - self.t_start) * progress, 
                  self.min_temperature)
        
        # Stability check: if entropy dangerously low, increase temperature
        if h_raw < 0.2 and tau < 1.0:
            tau = min(tau * 1.5, 1.0)

        adjusted = logits / tau
        h_adj = compute_entropy(adjusted)

        self.tracker.target_entropies.append(float("nan"))
        self.tracker.adjusted_entropies.append(h_adj)
        self.tracker.temperatures.append(tau)

        if scores.dim() > 1:
            return scores / tau
        return adjusted.unsqueeze(0)


class PassthroughProcessor(LogitsProcessor):
    """Baseline: no modification. Used with repetition_penalty in generate()."""

    def __init__(self, tracker: EntropyTracker):
        self.tracker = tracker

    def __call__(self, input_ids, scores):
        self.tracker.step += 1
        logits = scores[0] if scores.dim() > 1 else scores

        h = compute_entropy(logits)
        self.tracker.raw_entropies.append(h)
        self.tracker.adjusted_entropies.append(h)
        self.tracker.target_entropies.append(float("nan"))
        self.tracker.temperatures.append(1.0)
        return scores
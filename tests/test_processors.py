"""Unit tests for Gabriel's Horn logits processors (no model required)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entropy import compute_entropy, find_temperature_for_entropy
from src.processors import (
    EntropyTracker,
    HornDecayProcessor,
    PassthroughProcessor,
    ConstantTemperatureProcessor,
)


def flat_logits(vocab: int = 100) -> torch.Tensor:
    """Uniform-ish logits: maximum entropy start."""
    return torch.zeros(1, vocab) + torch.randn(1, vocab) * 0.01


def peaked_logits(vocab: int = 100) -> torch.Tensor:
    l = torch.zeros(1, vocab)
    l[0, 0] = 10.0
    return l


class TestEntropy:
    def test_uniform_entropy_near_log2_vocab(self):
        # compute_entropy returns BITS (base 2)
        h = compute_entropy(torch.zeros(100))
        assert abs(h - torch.log2(torch.tensor(100.0)).item()) < 0.01

    def test_peaked_entropy_near_zero(self):
        h = compute_entropy(peaked_logits()[0])
        assert h < 0.2  # bits; p_max ~ 0.9955 leaves ~0.07 bits

    def test_temperature_search_hits_target(self):
        logits = torch.randn(200) * 2
        h_raw = compute_entropy(logits)
        target = h_raw / 2
        tau = find_temperature_for_entropy(logits, target)
        h_adj = compute_entropy(logits / tau)
        assert abs(h_adj - target) < 0.05


class TestHornDecay:
    def test_warmup_passthrough(self):
        t = EntropyTracker()
        proc = HornDecayProcessor(t, decay_fn="inverse", warmup_steps=5)
        scores = flat_logits()
        out = proc(torch.zeros(1, 1, dtype=torch.long), scores.clone())
        assert torch.allclose(out, scores), "warmup must not modify scores"

    def test_entropy_clamped_after_warmup(self):
        t = EntropyTracker()
        proc = HornDecayProcessor(t, decay_fn="inverse", warmup_steps=1,
                                  floor_entropy=0.4)
        ids = torch.zeros(1, 1, dtype=torch.long)
        proc(ids, flat_logits())          # step 1: warmup, sets H1
        out = proc(ids, flat_logits())    # step 2: constraint binds
        h_out = compute_entropy(out[0])
        h_target = t.target_entropies[-1]
        assert h_out <= t.raw_entropies[-1], "entropy must not increase"
        assert abs(h_out - h_target) < 0.1 or h_out <= h_target + 0.1

    def test_targets_monotone_nonincreasing(self):
        t = EntropyTracker()
        proc = HornDecayProcessor(t, decay_fn="inverse", warmup_steps=1,
                                  floor_entropy=0.0)
        ids = torch.zeros(1, 1, dtype=torch.long)
        for _ in range(10):
            proc(ids, flat_logits())
        targets = t.target_entropies[1:]  # skip warmup entry
        assert all(a >= b - 1e-6 for a, b in zip(targets, targets[1:])), \
            "1/x targets must be non-increasing"

    def test_floor_respected(self):
        t = EntropyTracker()
        proc = HornDecayProcessor(t, decay_fn="inverse", warmup_steps=1,
                                  floor_entropy=0.4)
        ids = torch.zeros(1, 1, dtype=torch.long)
        for _ in range(50):
            proc(ids, flat_logits())
        assert min(t.target_entropies[1:]) >= 0.4 - 1e-6

    def test_state_offset_binds_immediately(self):
        """A2 perturbation mechanism: pre-advanced tracker → immediate clamp."""
        t = EntropyTracker()
        t.step = 60
        proc = HornDecayProcessor(t, decay_fn="inverse", warmup_steps=2,
                                  floor_entropy=0.4)
        ids = torch.zeros(1, 1, dtype=torch.long)
        # Structured logits: temperature CAN sharpen these (uniform ones it cannot)
        logits = torch.randn(1, 100) * 2
        h_raw = compute_entropy(logits[0])
        out = proc(ids, logits)  # step 61 > warmup: constraint active immediately
        h_out = compute_entropy(out[0])
        assert h_out < h_raw * 0.5,             "corrupted state must clamp entropy from the first step"


class TestBaselines:
    def test_passthrough_identity(self):
        t = EntropyTracker()
        proc = PassthroughProcessor(t)
        scores = flat_logits()
        out = proc(torch.zeros(1, 1, dtype=torch.long), scores.clone())
        assert torch.allclose(out, scores)

    def test_constant_temperature_scales(self):
        t = EntropyTracker()
        proc = ConstantTemperatureProcessor(t, temperature=0.5)
        scores = flat_logits()
        out = proc(torch.zeros(1, 1, dtype=torch.long), scores.clone())
        assert torch.allclose(out, scores / 0.5, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

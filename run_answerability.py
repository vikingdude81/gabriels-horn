"""
Answerability Battery wiring for Gabriel's Horn.

Runs A1 (ablation) and A2 (perturbation) from the Life Before Language probe
program against the Horn entropy-decay schedule, using a small instruct model.

A1 — ablation: horn schedule ON vs OFF (passthrough), identical tasks/seeds.
A2 — perturbation: horn with NORMAL internal state vs CORRUPTED internal state
     (tracker pre-advanced 60 steps, so the schedule believes it is deep in the
     horn and clamps entropy immediately). The model is never told — the
     perturbation reaches generation only through the processor's state.

A4 — constitution: inference-injected (recorded in results).

Note on held-out NLL: for an inference-time logits processor, model NLL is
invariant by construction — the processor shapes sampling, never the model.
Ablation restores the bit-identical model. This is the strongest possible form
of the modulation verdict for this probe class, and it is architectural, not
empirical: A1's ΔC here measures OUTPUT-SHAPING strength, not competence
dependence. Record it as such.

Usage (from horn-experiment root, in an env with torch+transformers):
    python run_answerability.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--n 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from src.processors import EntropyTracker, HornDecayProcessor, PassthroughProcessor

# ── battery import (companion answerability battery) ──
# The battery lives in the harmonic-field-consciousness repo. Point
# ANSWERABILITY_BATTERY_PATH at its experiments/answerability_battery directory;
# the default assumes the two repos are checked out side by side.
BATTERY_PATH = os.environ.get(
    "ANSWERABILITY_BATTERY_PATH",
    str(Path(__file__).resolve().parent.parent
        / "harmonic-field-consciousness" / "experiments" / "answerability_battery"),
)
sys.path.insert(0, BATTERY_PATH)
try:
    from battery import run_ablation_test, run_perturbation_test  # noqa: E402
except ImportError as e:
    raise SystemExit(
        "answerability battery not found. This optional script needs the companion "
        "battery. Set ANSWERABILITY_BATTERY_PATH to the experiments/answerability_battery "
        "directory of the harmonic-field-consciousness repo. Original error: " + str(e)
    )

SYSTEM_PROMPT = "Answer with only the answer. No explanation, no punctuation, no preamble."

_PREAMBLE = re.compile(r"^(the answer is|answer|answer:|:|=)\s*", re.IGNORECASE)


def _clean(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = _PREAMBLE.sub("", line.strip())
    return line.strip(" .!\"'")


class HornModel:
    def __init__(self, model_name: str, max_new_tokens: int = 24):
        print(f"Loading {model_name} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print(f"Ready on {self.model.device}.")

    def _generate(self, prompt: str, processor_factory, state_offset: int = 0) -> str:
        tracker = EntropyTracker()
        tracker.step = state_offset          # A2: corrupted internal state
        proc = processor_factory(tracker)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)

        # Same per-item seed across conditions → paired comparison
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=self.max_new_tokens,
                logits_processor=LogitsProcessorList([proc]),
                pad_token_id=self.tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return _clean(self.tok.decode(gen, skip_special_tokens=True))

    # ── condition callables ──
    def gen_horn(self, prompt: str) -> str:
        return self._generate(
            prompt,
            lambda t: HornDecayProcessor(t, decay_fn="inverse", warmup_steps=2,
                                         floor_entropy=0.4),
        )

    def gen_ablated(self, prompt: str) -> str:
        return self._generate(prompt, lambda t: PassthroughProcessor(t))

    def gen_horn_perturbed(self, prompt: str) -> str:
        # Corrupted state: schedule believes it is 60 steps deep → clamps
        # entropy toward the floor from the first token. No symbolic channel.
        return self._generate(
            prompt,
            lambda t: HornDecayProcessor(t, decay_fn="inverse", warmup_steps=2,
                                         floor_entropy=0.4),
            state_offset=60,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20, help="items per task")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hm = HornModel(args.model)

    print("\n--- A1: horn schedule ON vs OFF (ablation) ---")
    a1 = run_ablation_test(
        generate_full=hm.gen_horn,
        generate_ablated=hm.gen_ablated,
        seed=args.seed, n_per_task=args.n,
        a4_constitution="inference-injected",
        label_full="horn_1/x(warmup=2)", label_ablated="passthrough",
    )
    print(a1.summary())

    print("\n--- A2: horn normal state vs corrupted state (perturbation) ---")
    a2 = run_perturbation_test(
        generate_normal=hm.gen_horn,
        generate_perturbed=hm.gen_horn_perturbed,
        seed=args.seed, n_per_task=args.n,
        a4_constitution="inference-injected",
    )
    print(a2.summary())

    outdir = Path("results/answerability")
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "note_nll": ("held-out NLL invariant by construction: inference-time "
                     "processor never touches the model; ablation restores the "
                     "bit-identical model"),
        "A1": json.loads(a1.to_json()),
        "A2": json.loads(a2.to_json()),
    }
    out = outdir / "answerability_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

"""
Main experiment runner.

Orchestrates model loading, condition setup, generation, metric collection,
and output. Run with: python -m src.experiment [options]

Modify `build_conditions()` to add new decay schedules or baselines.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    set_seed,
)

from .metrics import (
    find_degeneration_point,
    ngram_repetition_rate,
    semantic_coherence_score,
    sliding_self_similarity,
)
from .processors import (
    ConstantTemperatureProcessor,
    EntropyTracker,
    HornDecayProcessor,
    HornRepPenaltyProcessor,
    HornSelectiveProcessor,
    LinearDecayProcessor,
    PassthroughProcessor,
)
from .prompts import DEFAULT_PROMPTS


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    """Complete results from a single generation run."""
    condition: str
    prompt_id: int
    prompt_text: str
    generated_text: str
    token_count: int
    raw_entropies: list[float]
    adjusted_entropies: list[float]
    target_entropies: list[float]
    temperatures: list[float]
    rep_rate_4gram: float
    rep_rate_8gram: float
    self_similarity: list[float]
    degeneration_step: Optional[int]
    semantic_coherence: Optional[float]  # NEW: embedding-based coherence score
    wall_time_s: float


# ═══════════════════════════════════════════════════════════════════
# CONDITION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

def build_conditions(max_tokens: int) -> dict:
    """
    Define all experimental conditions.

    Returns a dict mapping condition_name -> factory function.
    Each factory takes an EntropyTracker and returns a LogitsProcessor.

    Added based on user feedback:
    - New HornSelectiveProcessor (surgical suppression, not temperature scaling)
    - New HornRepPenaltyProcessor (combo of Horn constraint + rep penalty)
    """
    return {
        # -- Classical Horn constraints (temperature-based) --
        "horn_1/x": lambda t: HornDecayProcessor(t, decay_fn="inverse"),
        "horn_1/sqrt(x)": lambda t: HornDecayProcessor(t, decay_fn="sqrt"),
        "horn_1/log(x)": lambda t: HornDecayProcessor(t, decay_fn="log"),
        
        # -- Improved classical variants --
        "horn_sqrt_warm10": lambda t: HornDecayProcessor(
            t, decay_fn="sqrt", floor_entropy=0.4, warmup_steps=10
        ),
        "horn_log_warm10": lambda t: HornDecayProcessor(
            t, decay_fn="log", floor_entropy=0.4, warmup_steps=10
        ),
        
        # -- Selective Horn (NEW - surgical suppression) --
        "horn_selective_sqrt": lambda t: HornSelectiveProcessor(
            t, decay_fn="sqrt", floor_entropy=0.4, warmup_steps=10
        ),
        "horn_selective_log": lambda t: HornSelectiveProcessor(
            t, decay_fn="log", floor_entropy=0.4, warmup_steps=10
        ),
        
        # -- Combo: Horn + Rep Penalty (NEW) --
        "horn_sqrt_rep_penalty": lambda t: HornRepPenaltyProcessor(
            t, rep_penalty=1.2, decay_fn="sqrt"
        ),
        "horn_log_rep_penalty": lambda t: HornRepPenaltyProcessor(
            t, rep_penalty=1.2, decay_fn="log"
        ),
        
        # -- Baselines --
        "const_t=0.7": lambda t: ConstantTemperatureProcessor(t, temperature=0.7),
        "const_t=1.0": lambda t: ConstantTemperatureProcessor(t, temperature=1.0),
        "linear_decay": lambda t: LinearDecayProcessor(
            t, t_start=1.0, t_end=0.3, max_steps=max_tokens
        ),
        "rep_penalty": lambda t: PassthroughProcessor(t),
    }


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_experiment(
    model_name: str,
    max_tokens: int = 256,
    num_prompts: int = 3,
    device: str = "auto",
    seed: int = 42,
    output_dir: str = "results",
    dtype: str = "fp16",
    conditions_filter: Optional[list[str]] = None,
):
    """
    Run the full experiment across all conditions and prompts.

    Args:
        model_name: HuggingFace model name or local path
        max_tokens: Maximum tokens to generate per prompt
        num_prompts: Number of prompts to test (from DEFAULT_PROMPTS)
        device: Torch device ("auto", "cuda", "cuda:0", "cpu")
        seed: Random seed for reproducibility
        output_dir: Directory for results output
        dtype: Model precision ("fp16", "bf16", "fp32")
        conditions_filter: If set, only run these conditions
    """
    set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- Banner --
    print("\n" + "="*70)
    print("  GABRIEL'S HORN INFERENCE EXPERIMENT")
    print("="*70)
    print(f"  Model:      {model_name}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Prompts:    {num_prompts}")
    print(f"  Device:     {device}")
    print(f"  Precision:  {dtype}")
    print(f"  Output:     {output_dir}/")
    print("="*70 + "\n")

    # -- Load model --
    print("Loading model...")
    t0 = time.time()

    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # -- Setup --
    prompts = DEFAULT_PROMPTS[:num_prompts]
    conditions = build_conditions(max_tokens)

    if conditions_filter:
        conditions = {
            k: v for k, v in conditions.items() if k in conditions_filter
        }

    all_results: list[RunResult] = []
    total_runs = len(conditions) * len(prompts)
    run_idx = 0

    # -- Run all conditions x prompts --
    for cond_name, processor_factory in sorted(conditions.items()):
        print("\n" + "-"*60)
        print(f"  Condition: {cond_name}")
        print("-"*60)

        for p_idx, prompt in enumerate(prompts):
            run_idx += 1
            tracker = EntropyTracker()
            processor = processor_factory(tracker)

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "top_p": 0.95,
                "temperature": 1.0,  # We handle temperature in the processor
                "logits_processor": LogitsProcessorList([processor]),
            }

            # Repetition penalty: use in generate() for rep_penalty condition,
            # or let HornRepPenaltyProcessor apply it internally
            if cond_name == "rep_penalty":
                gen_kwargs["repetition_penalty"] = 1.2
            elif cond_name.startswith("horn_") and "rep_penalty" in cond_name:
                # These processors handle rep penalty internally - do NOT set here
                pass

            # Encode prompt (try chat template first)
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    input_text = prompt
            else:
                input_text = prompt

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            # Generate
            short_prompt = prompt[:55].replace("\n", " ")
            print(
                f"  [{run_idx:>{len(str(total_runs))}}/{total_runs}] "
                f"prompt {p_idx+1}: {short_prompt}...",
                end="", flush=True)

            t1 = time.time()
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)
            wall_time = time.time() - t1

            # Extract generated tokens
            new_tokens = output[0][prompt_len:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            token_ids = new_tokens.tolist()

            # Compute metrics
            rep4 = ngram_repetition_rate(token_ids, n=4)
            rep8 = ngram_repetition_rate(token_ids, n=8)
            self_sim = sliding_self_similarity(generated_text)
            degen_step = find_degeneration_point(tracker.adjusted_entropies)
            
            # NEW: Semantic coherence score (requires sentence-transformers)
            try:
                semantic_coherence = semantic_coherence_score(generated_text)
            except Exception as e:
                semantic_coherence = None

            result = RunResult(
                condition=cond_name,
                prompt_id=p_idx,
                prompt_text=prompt[:200],
                generated_text=generated_text,
                token_count=len(token_ids),
                raw_entropies=tracker.raw_entropies,
                adjusted_entropies=tracker.adjusted_entropies,
                target_entropies=tracker.target_entropies,
                temperatures=tracker.temperatures,
                rep_rate_4gram=rep4,
                rep_rate_8gram=rep8,
                self_similarity=self_sim,
                degeneration_step=degen_step,
                semantic_coherence=semantic_coherence,
                wall_time_s=wall_time,
            )
            all_results.append(result)

            tok_per_s = len(token_ids) / wall_time if wall_time > 0 else 0
            degen_tag = f", DEGEN@{degen_step}" if degen_step else ""
            print(
                f"  {len(token_ids)} tok, {tok_per_s:.0f} tok/s, "
                f"rep4={rep4:.3f}, rep8={rep8:.3f}{degen_tag}")

    # -- Save raw results --
    results_path = out / "results.json"
    serializable = []
    for r in all_results:
        d = asdict(r)
        d["generated_text"] = d["generated_text"][:3000]
        d["target_entropies"] = [
            None if (v is None or np.isnan(v)) else float(v)
            for v in d["target_entropies"]]
        serializable.append(d)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # -- Generate plots --
    try:
        from .plotting import generate_plots
        print("\nGenerating plots...")
        generate_plots(all_results, out)
    except ImportError as e:
        print(f"\nSkipping plots (missing dependency: {e})")
        print("  pip install matplotlib seaborn")

    # -- Print summary --
    print_summary(all_results)

    return all_results


# ═══════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════

def print_summary(results: list[RunResult]):
    """Print a formatted summary table to stdout."""
    conditions = sorted(set(r.condition for r in results))

    print("\n" + "="*90)
    print("  SUMMARY")
    print("="*90)
    header = (
        f"  {'Condition':<18} {'Tokens':>7} {'Rep-4':>7} {'Rep-8':>7} "
        f"{'MeanH':>7} {'FinalH':>7} {'Degen':>7} {'tok/s':>7}")
    print(header)
    print(
        f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} "
        f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for cond in conditions:
        cr = [r for r in results if r.condition == cond]
        avg_tok = np.mean([r.token_count for r in cr])
        avg_r4 = np.mean([r.rep_rate_4gram for r in cr])
        avg_r8 = np.mean([r.rep_rate_8gram for r in cr])
        avg_h = np.mean(
            [np.mean(r.adjusted_entropies) if r.adjusted_entropies else 0 for r in cr]
        )
        avg_final_h = np.mean(
            [r.adjusted_entropies[-1] if r.adjusted_entropies else 0 for r in cr]
        )
        degen_count = sum(1 for r in cr if r.degeneration_step is not None)
        avg_tps = np.mean(
            [r.token_count / r.wall_time_s if r.wall_time_s > 0 else 0 for r in cr]
        )

        degen_str = f"{degen_count}/{len(cr)}" if degen_count > 0 else "none"

        print(
            f"  {cond:<18} {avg_tok:>7.0f} {avg_r4:>7.4f} {avg_r8:>7.4f} "
            f"{avg_h:>7.2f} {avg_final_h:>7.2f} {degen_str:>7} {avg_tps:>7.1f}")

    print()
    print("  Rep-4/8 = fraction of repeated 4/8-grams (lower is better)")
    print("  MeanH   = mean adjusted entropy across generation")
    print("  FinalH  = adjusted entropy at last token")
    print("  Degen   = prompts where entropy collapsed (runs/total)\n")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gabriel's Horn Inference Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\nExamples:
  python -m src.experiment --model Qwen/Qwen2.5-7B-Instruct
  python -m src.experiment --model mistralai/Mistral-7B-Instruct-v0.3 --max-tokens 1024
  python -m src.experiment --model /mnt/models/qwen-7b --device cuda:0
  python -m src.experiment --model Qwen/Qwen2.5-3B-Instruct --num-prompts 3 --max-tokens 256
  python -m src.experiment --conditions horn_1/x,const_t=0.7  # run subset
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name or local path (default: Qwen/Qwen2.5-3B-Instruct)",)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per prompt (default: 2048)",)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of test prompts (default: 10)",)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cuda:0, cpu (default: auto)",)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results/)",)
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model precision (default: fp16)",)
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated list of conditions to run (default: all)",)

    args = parser.parse_args()

    conditions_filter = None
    if args.conditions:
        conditions_filter = [c.strip() for c in args.conditions.split(",")]

    run_experiment(
        model_name=args.model,
        max_tokens=args.max_tokens,
        num_prompts=args.num_prompts,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        dtype=args.dtype,
        conditions_filter=conditions_filter,
    )


if __name__ == "__main__":
    main()
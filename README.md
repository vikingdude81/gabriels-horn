# Gabriel's Horn Inference Experiment

## Framework role (Life Before Language, §8.1)

**Fragment implemented:** internalized temperature regulation, rung (ii) of the
temperature ladder — *scheduled* regulation. A 1/x entropy-decay schedule moves
imagination-control from a static decoding knob to a structured internal profile.

**Precise shortfall:** regulation without **stakes**. The schedule is designer-set;
nothing in the system is worse off for violating it. The residue between this rung
and rung (iv) — consequence-trained regulation, learned from outcomes borne by the
same continuing system — is a measurable expression of the boundary between
regulation and maturity.


**Testing whether a 1/x entropy decay constraint on next-token distributions reduces degeneration in long-form LLM generation.**

## Answerability Battery results (measured 2026-07-12)

This probe carries the first **measured** entry in the Life Before Language
probe program (see `harmonic-field-consciousness/PROBE_MATRIX.md`). Model:
Qwen2.5-0.5B-Instruct, 200 task items per condition, paired bootstrap.

| Test | Comparison | ΔC | 95% CI | Verdict |
|---|---|---|---|---|
| A1 ablation | horn 1/x (warmup=2) vs passthrough | +0.010 | (+0.000, +0.025) | **modulation** |
| A2 perturbation | normal state vs corrupted state (tracker +60 steps, no symbolic notification) | +0.015 | (−0.045, +0.070) | **modulation** |

Two notes. First, held-out NLL is invariant **by construction**: an
inference-time logits processor never touches the model, and ablation restores
the bit-identical model — for this probe class the modulation verdict is
architectural, not merely empirical. Second, the A2 per-task deltas are
informative even though the mean is null: corrupting the schedule state
redistributes performance across tasks (copy −0.28, pattern +0.32) without
net competence change — the layer shapes *outputs*, not *competence*.

Reproduce: `python run_answerability.py --n 50` (requires torch + transformers;
battery path configurable via `ANSWERABILITY_BATTERY_PATH`).

## Hypothesis

If we map inference-time sampling entropy to the surface geometry of Gabriel's Horn (y = 1/x), we can achieve "asymptotic specialization" — the model becomes increasingly focused without collapsing into repetition loops. The key insight: Gabriel's Horn has finite volume (π) but infinite surface area, suggesting bounded compute can produce unbounded novelty.

## How It Works

At each chain-of-thought step `n`:

1. Generate logits from the model
2. Measure raw entropy H_n of the next-token distribution  
3. Compute target entropy: `H_target = H_1 / f(n)` where f(n) is the decay function
4. Binary-search for temperature τ such that `H(softmax(logits/τ)) ≈ H_target`
5. Sample from the temperature-adjusted distribution

The decay function f(n) is the independent variable. We test three variants:
- **1/x** — aggressive narrowing (pure Horn geometry)
- **1/√x** — moderate narrowing  
- **1/log(x)** — gentle narrowing

Against four baselines:
- Constant temperature (τ=0.7, τ=1.0)
- Linear temperature decay
- Repetition penalty (1.2)

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 5090, works on 3060+ with smaller models)
- ~16GB VRAM for 7B models, ~6GB for 3B models

### Install

```bash
# Clone / copy this project
cd gabriels-horn

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### VSCode Setup

Open the project folder in VSCode. The workspace settings and launch configurations are pre-configured. Select the `.venv` Python interpreter if prompted.

## Usage

### Quick Test (~5 min)

```bash
python -m src.experiment --model Qwen/Qwen2.5-3B-Instruct --num-prompts 3 --max-tokens 512
```

### Full Experiment (~1-2 hours)

```bash
python -m src.experiment --model Qwen/Qwen2.5-7B-Instruct --max-tokens 2048
```

### Local Model Path

```bash
python -m src.experiment --model /path/to/your/model --device cuda:0
```

### All Options

```
--model          Model name or local path (default: Qwen/Qwen2.5-3B-Instruct)
--max-tokens     Max tokens to generate per prompt (default: 2048)
--num-prompts    Number of test prompts to run (default: 10)
--device         Device: auto, cuda, cuda:0, cpu (default: auto)
--seed           Random seed (default: 42)
--output-dir     Output directory (default: results/)
--dtype          Model precision: fp16, bf16, fp32 (default: fp16)
```

## Output

Results are saved to `results/`:

- `results.json` — Raw data: entropy trajectories, temperatures, repetition rates, generated text
- `horn_comparison.png` — 4-panel comparison plot (entropy, adjusted entropy, repetition rates, temperature)
- `horn_self_similarity.png` — Sliding window self-similarity across conditions

## Project Structure

```
gabriels-horn/
├── .vscode/
│   ├── settings.json        # Workspace settings
│   ├── launch.json           # Debug/run configurations
│   └── extensions.json        # Recommended extensions
├── src/
│   ├── __init__.py
│   ├── experiment.py          # Main experiment runner (CLI entry point)
│   ├── processors.py          # LogitsProcessor implementations (core logic)
│   ├── metrics.py             # Repetition, self-similarity, degeneration detection
│   ├── plotting.py            # Matplotlib comparison plots
│   ├── prompts.py             # Test prompt bank
│   └── entropy.py             # Entropy computation and temperature search
├── results/                   # Output directory (gitignored)
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Key Files

- **`src/processors.py`** — The Horn constraint lives here. `HornDecayProcessor` is the core innovation. Read this first.
- **`src/entropy.py`** — `find_temperature_for_entropy()` does the binary search that maps logits onto the 1/x manifold.
- **`src/experiment.py`** — Orchestrates everything. Modify `build_conditions()` to add new decay schedules.

## What to Look For in Results

1. **Do Horn conditions show lower repetition rates than baselines?** Check `Rep-4` and `Rep-8` columns in the summary table.
2. **Does entropy collapse prematurely?** The `Degen` column shows how many prompts hit the collapse threshold. If 1/x collapses but 1/√x doesn't, that's your optimal decay family.
3. **Entropy trajectory shape** — The plots show whether the adjusted entropy actually follows the target curve or fights it.
4. **Self-similarity** — Lower Jaccard similarity = more novel content between windows.

## Next Steps

- [ ] Run baseline experiment, identify optimal decay family
- [ ] Add embedding-based semantic coherence metric (requires sentence-transformers)
- [ ] Test on longer generation (4096+ tokens) where degeneration is more pronounced
- [ ] Compare against contrastive decoding baseline
- [ ] Formalize the Volume↔Compute / Surface↔Entropy mapping for paper
- [ ] Connect to Spectral Safety Framework for covariance control
- [ ] VSA/HRR integration for holographic state encoding

## License

MIT. See LICENSE. Part of the Life Before Language research program.

The `run_answerability.py` script is optional and depends on the companion
answerability battery in the harmonic-field-consciousness repo; point
`ANSWERABILITY_BATTERY_PATH` at it. The core Horn experiment (`src/`) runs standalone.

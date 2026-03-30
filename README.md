# HydraLM

**An adaptive evolutionary language model with a living cell population.**

*genomorphic · in collaboration with Claude Sonnet 4.6 (Anthropic)*

---

## What is this?

HydraLM is a language model that evolves its own internal structure during training. Instead of a fixed architecture, it maintains a **population of adaptive cells** organised as a directed acyclic graph (DAG). Cells are born, compete for survival based on their contribution to learning, donate their weights to offspring, and die — all while the model is training on text.

The goal is to explore whether a model that can restructure itself is more sample-efficient and more interpretable than one with a fixed architecture of the same parameter count.

This is active research code. It works, it trains, and it produces interesting dynamics. It is not a polished library.

---

## Key Ideas

### Evolutionary Cell Population
The model maintains between 3 and 24 adaptive cells. Each cell is a small feedforward unit with its own health score, age, and contribution history. A tournament-style evolutionary algorithm (RelativeEA) runs every N steps:
- The bottom-scoring cells are pruned gradually (max 7 per round, not all at once)
- New cells are spawned from elite parents via crossover
- At least one cell per DAG layer is always preserved

### Gene Pool
High-contribution cells donate their weights to a persistent ring buffer (200 snapshots) when they are pruned or frozen. New cells can inherit from this pool via three modes chosen at random:
- **Direct** — truncate/pad from the closest-shape historical donor
- **Average** — average the overlap region across all same-layer donors  
- **Subspace** — SVD the best donor and project its principal directions into the child's weight space

This gives new cells a head start from patterns learned much earlier in training, not just from the current population.

### Shapley Health Attribution
Cell health is computed using path-aware permutation Shapley values over the full DAG. This means a deep cell gets credit for what it contributes *through* its downstream consumers, not just its direct output. Cells that stop contributing lose health and tenure over time.

### Homeostasis
Four internal scalars drive intrinsic training rewards:
- **Energy** `[0,1]` — drains each step, recovers when loss is below baseline. Low energy increases curiosity, upweighting hard tokens.
- **Integrity** `[0,1]` — tracks CLM improvement trend. Low integrity forces deeper processing via the uncertainty gate.
- **Excitement** `[0,1]` — spikes on large loss changes. High excitement adds entropy regularisation.
- **Pain** `[0,1]` — acute distress when loss exceeds baseline significantly. High pain clips gradients from the hardest tokens to prevent destabilisation.

### Uncertainty Gate
A learned multi-stage gate routes tokens through deeper processing only when uncertain. The depth of processing is modulated by the integrity homeostasis signal.

### Autoregressive Loops
The model can run the cell population multiple times per forward pass (up to `max_loops`). An exit gate learns when additional loops stop being useful, penalised by a compute tax.

---

## Files

| File | Purpose |
|------|---------|
| `hydra_lm.py` | Full model: tokenizer, encoder, cell population, EA, homeostasis, gene pool |
| `train.py` | Streaming trainer with checkpointing and resume |
| `prepare_data.py` | Downloads a subset of The Pile and tokenizes it |
| `inspect_checkpoint.py` | Detailed checkpoint inspector — cell health, Shapley trends, homeostasis, DAG wiring |

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch tqdm datasets huggingface_hub
```

### 2. Prepare data
```bash
python prepare_data.py --output_dir data/ --n_docs 100000
```

### 3. Train
```bash
python train.py \
    --data_path data/corpus.txt \
    --run_dir   runs/run1 \
    --vocab_size 8000 \
    --d_model   256 \
    --max_cells 24 \
    --steps     100000    
```
Run `python train.py --help` for the full list of arguments and their defaults.

### 4. Resume from checkpoint
```bash
python train.py \
    --data_path data/corpus.txt \
    --run_dir   runs/run1 \
    --resume
```

### 5. Inspect a checkpoint
```bash
python inspect_checkpoint.py runs/run1/ckpt_step00010000.json
```

---

## Checkpoint Format

Each checkpoint saves four files:

```
ckpt_step00010000.json       # Human-readable topology, config, EA state
ckpt_step00010000.pt         # Weights and buffers (companion to JSON)
ckpt_step00010000_genepool.pt # Gene pool snapshots
ckpt_opt_00010000.pt         # Optimizer and scheduler state (resume only)
ckpt_latest.txt              # Pointer to most recent checkpoint base name
```

The `.json` + `.pt` pair is the portable DNA — enough to reconstruct the exact evolved state on any hardware. The optimizer file is only needed to resume training.

To roll back to an earlier checkpoint manually:
```bash
echo -n "ckpt_step00008000" > runs/run1/ckpt_latest.txt
```

---

## Architecture Overview

```
Token IDs
    │
    ▼
TransformerEncoder (contextual embeddings)
    │
    ▼
Homeostasis vector (energy, integrity, excitement, pain)
    │
    ├──► Cell 0 (L0, root) ──────────────────────┐
    ├──► Cell 1 (L0, root) ──────────────────────┤
    ├──► Cell 2 (L0, root) ──────────────────────┤
    │                                             │
    ├──► Cell 6 (L1) ◄── [Cell 0, 1, 2, ...]    ├──► Aggregate
    │                                             │         │
    └──► Cell 12 (L2) ◄── [Cell 6, ...]  ────────┘         │
                                                            ▼
                                                    LanguageHead
                                                            │
                                                            ▼
                                                    Next-token logits
                                                    (× max_loops)
```

The DAG topology evolves. At any checkpoint, `inspect_checkpoint.py` shows the exact wiring.

---

## Inspector Output

```
  CELL POPULATION  (17 cells)

  ── Layer distribution ──
  L 0 : ●●●●●●●●●●●●●●  (14 cells)
  L 1 : ●●  (2 cells)
  L 2 : ●  (1 cell)

  ── Cell details ──
   ID  L   W_in  W_out  Srcs   Health        Age   Tenure  Frz
    6  1   4868    135     9    0.821  ██████████  12400    0.043   no
   12  2   1796    420     3    0.744  ███████░░░   9820    0.031   no
    0  0    260    482     0    0.690  █████████░   15200    0.028   no
  ...
```

---

## Current Status

- Training is stable across multi-thousand-step runs
- DAG depth grows naturally (L0 → L1 → L2) 
- Homeostasis signals are active and modulating training
- Shapley attribution is flowing correctly to the stats tracker
- Gene pool is filling and flashing to new cells
- Loss at step ~2000: ~8.4 (8k vocab, small corpus, early training)

Known limitations:
- All cells currently L0 in early training — deeper layers take time to establish
- Loop gate learns slowly; most steps still use max loops at step 1250
- Tenure normalisation aggressive — lifetime_contrib near zero across population

---

## Hyperparameters

Key parameters and their defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 8000 | BPE vocabulary size |
| `d_model` | 256 | Transformer hidden dimension |
| `n_layers` | 4 | Transformer encoder layers |
| `max_cells` | 24 | Maximum cell population size |
| `n_dag_layers` | 10 | Maximum DAG depth |
| `max_connections` | 10 | Maximum sources per cell |
| `growth_interval` | 200 | Steps between EA evaluations |
| `max_loops` | 4 | Maximum recurrent passes |
| `freeze_age` | 800 | Minimum age before a cell can freeze |

---

## Design Decisions and Rationale

**Why gradual pruning?** Early runs showed a boom-bust cycle: population hits max_cells, all non-elite cells are pruned at once (19 of 24), deep structure collapses to 5–7 shallow cells. Capping pruning at 7 cells per round prevents this.

**Why a gene pool instead of just elite crossover?** Live elite donors have matching in_features only when cells happen to have identical wiring. In a DAG where each cell's input dimension depends on its sources, exact shape matches are rare. The gene pool stores historical snapshots and uses truncate/pad or SVD projection to make inheritance work across shape mismatches.

**Why positive pain?** Pain was originally modelled as a negative RL reward in `[-1, 0]`. The other three homeostasis scalars are positive `[0, 1]`. The sign mismatch meant the `intrinsic_rewards()` function had to `abs()` pain before using it — a code smell. Pain is now a distress level in `[0, 1]` consistent with the rest of the system.

**Why 50% elite for crossover?** The original 20% elite meant only ~5 cells at max population were crossover-eligible. With DAG diversity, those 5 rarely had matching shapes, making crossover nearly impossible. 50% gives 12 eligible donors and a much higher chance of useful inheritance.

---

## Acknowledgements

This project has been a genuinely collaborative research process across 
multiple AI systems, each contributing in different ways:

- **ChatGPT** — suggested Hebbian plasticity as a learning mechanism, 
  which shaped the cell plasticity implementation
- **Gemini** — helped refine and articulate the homeostasis concept into 
  something architecturally coherent
- **Grok** — contributed to shaping the homeostasis framework and its 
  four-scalar design
- **DeepSeek** — ongoing analysis of training runs and checkpoint states, 
  informing research direction between sessions
- **Claude Sonnet 4.6** — primary implementation, architecture decisions, 
  and code throughout the project

The human researcher directed the vision, interpreted results, and made 
all conceptual and architectural decisions. The AI systems above served 
as a thinking environment for developing and stress-testing those ideas.

---

## License

MIT

---

## Citation

If you build on this work:

```
@misc{hydralm2026,
  author = {genomorphic and Claude Sonnet 4.6 (Anthropic)},
  title  = {HydraLM: Adaptive Evolutionary Language Model},
  year   = {2026},
  url    = {https://github.com/genomorphic/hydra-lm}
}
```

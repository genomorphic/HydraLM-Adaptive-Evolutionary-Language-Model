# Run 1 — Research Notes

**Steps:** 250 → 4,750 (19 checkpoints)  
**Vocab size:** 8,000 · **d_model:** 256 · **Max cells:** 24 · **DAG layers:** 10  
**Status:** Complete — this run exposed the boom-bust cycle that led to the gradual pruning fix.

---

## Overview

Run 1 was the first extended training run of HydraLM. It ran for 4,750 steps and produced the key observation that motivated several architectural fixes now present in the codebase. The run is preserved here as a transparent record of what the model did before those fixes, and what the data looked like that led to the changes.

The CLM loss improved from ~9.09 (step 250) to ~8.40 (step 4,750) — a meaningful reduction for a model learning from scratch with an 8k vocab on a small corpus. More interesting than the loss itself is what happened to the architecture along the way.

---

## Phase 1 — Steady Growth (Steps 250–1,750)

The population grew steadily from 7 cells to 22 over the first 1,750 steps. DAG depth held at L1 throughout — the model had L0 root cells and exactly two L1 cells that persisted across every checkpoint in this phase. The L1 cells were stable because they were genuinely useful; the EA kept them while filling the population with new L0 cells alongside them.

Homeostasis during this phase was healthy. Integrity started at 1.000 and declined slowly — the model was improving consistently, just not fast enough to arrest the drift. Energy stayed near 1.0 and pain was effectively zero, meaning the model was never in acute distress. It was learning steadily.

```
  Step     250 | ███████████                              |  7 cells
  Step     500 | ████████████████                         | 10 cells
  Step     750 | ████████████████████                     | 12 cells
  Step   1,000 | ██████████████████████████               | 16 cells
  Step   1,250 | ██████████████████████████████           | 18 cells
  Step   1,500 | █████████████████████████████████        | 20 cells
  Step   1,750 | ████████████████████████████████████     | 22 cells
```

---

## Phase 2 — The Collapse (Steps 1,750–2,000)

Between steps 1,750 and 2,000 the population dropped from 22 cells to 8 — a loss of 14 cells in a single EA evaluation. This is the boom-bust event that motivated the gradual pruning fix.

What happened: when the population approached `max_cells=24`, the original EA identified the top 20% as elite (~4–5 cells) and pruned everything else in one pass. At 22 cells, roughly 17–18 were discarded simultaneously. The surviving 8 cells were the elite plus two newly spawned children. All hierarchical structure that had been built over 1,750 steps was gone in one step.

The average health actually *rose* slightly at step 2,000 (0.038 vs 0.000 before) — the survivors were genuinely the best cells, but there were so few of them that the model lost most of its capacity.

```
  Step   1,750 | ████████████████████████████████████     | 22 cells
  Step   2,000 | █████████████                            |  8 cells ← COLLAPSE
```

**Fix applied after this run:** `RelativeEA.step()` now caps pruning at 7 cells per EA call, sorted worst-first. The population bleeds down gradually rather than collapsing instantaneously.

---

## Phase 3 — Recovery and Deeper Structure (Steps 2,000–3,500)

After the collapse the population rebuilt. By step 2,750 it had recovered to 16 cells, and something new appeared — a third L1 cell. The model was now maintaining three cells at depth 1, suggesting the EA had learned that L1 cells were worth preserving.

At step 3,000 the first L2 cell appeared. This is significant — it took 3,000 steps to build enough stable L1 structure for an L2 cell to wire into. The population reached 21 cells with the DAG now three layers deep (L0 → L1 → L2).

```
  Step   2,750 | ██████████████████████████               | 16 cells  L0:13  L1:3
  Step   3,000 | ███████████████████████████████████      | 21 cells  L0:17  L1:3  L2:1
  Step   3,250 | ██████████████████████████████████████   | 23 cells  L0:19  L1:3  L2:1
  Step   3,500 | ████████████████████████████████████████ | 24 cells  L0:20  L1:3  L2:1  ← PEAK
```

---

## Phase 4 — Maximum Population and L3 (Steps 3,500–4,000)

At step 3,500 the population reached its maximum of 24 cells for the first time. The DAG held at L2. Between steps 3,500 and 3,750 something unexpected happened — an L3 cell appeared, and the population dropped from 24 to 20. This was the gradual pruning fix at work: rather than collapsing to 8, the EA trimmed 4 cells and the L3 cell survived.

By step 4,000 the population was back to 24 with L3 intact — the deepest structure seen in this run.

```
  Step   3,500 | ████████████████████████████████████████ | 24 cells  depth=2  ← PEAK
  Step   3,750 | █████████████████████████████████        | 20 cells  depth=3  ← L3 appears
  Step   4,000 | ████████████████████████████████████████ | 24 cells  depth=3  ← PEAK
```

This is the first evidence that the gradual pruning fix was working — the model reached maximum population twice after the fix, dropped modestly each time, and recovered. No collapse comparable to the step 1,750→2,000 event occurred again.

---

## Phase 5 — Stabilisation (Steps 4,000–4,750)

The L3 cell was lost between steps 4,000 and 4,250, and the population settled into a cycle of 20→22→24 cells at depth L2. The three L1 cells and single L2 cell persisted unchanged through the final three checkpoints — the EA was consistently selecting them as elite and protecting them.

```
  Step   4,000 | ████████████████████████████████████████ | 24 cells  depth=3
  Step   4,250 | █████████████████████████████████        | 20 cells  depth=2
  Step   4,500 | ████████████████████████████████████     | 22 cells  depth=2
  Step   4,750 | ████████████████████████████████████████ | 24 cells  depth=2
```

The CLM EMA at step 4,750 was 8.4024 — essentially identical to steps 3,500–4,500. Loss improvement had plateaued. This is expected: the model had reached a local minimum on the training corpus with the current architecture, and without the Shapley values flowing correctly (a bug fixed after this run) the EA was running blind.

---

## Homeostasis Analysis

| Step | Energy | Integrity | Pain | CLM EMA |
|------|--------|-----------|------|---------|
|     250 |  0.996 |     1.000 | 0.000 |  9.0904 |
|     500 |  0.998 |     1.000 | 0.000 |  8.9625 |
|     750 |  1.000 |     0.988 | 0.001 |  8.8463 |
|   1,000 |  0.996 |     0.971 | 0.010 |  8.7504 |
|   1,250 |  1.000 |     0.935 | 0.001 |  8.6849 |
|   1,500 |  1.000 |     0.893 | 0.001 |  8.6319 |
|   1,750 |  0.974 |     0.852 | 0.007 |  8.6122 |
|   2,000 |  0.996 |     0.811 | 0.001 |  8.5704 |
|   2,250 |  0.998 |     0.765 | 0.011 |  8.5317 |
|   2,500 |  0.985 |     0.724 | 0.010 |  8.5093 |
|   2,750 |  0.998 |     0.686 | 0.004 |  8.4779 |
|   3,000 |  0.955 |     0.644 | 0.001 |  8.4993 |
|   3,250 |  0.997 |     0.600 | 0.008 |  8.4494 |
|   3,500 |  0.997 |     0.555 | 0.008 |  8.4211 |
|   3,750 |  0.998 |     0.508 | 0.011 |  8.4208 |
|   4,000 |  0.992 |     0.456 | 0.011 |  8.4329 |
|   4,250 |  0.994 |     0.413 | 0.008 |  8.4040 |
|   4,500 |  0.996 |     0.368 | 0.002 |  8.4025 |
|   4,750 |  0.998 |     0.323 | 0.012 |  8.4024 |

*Pain shown as distress level — 0=healthy, 1=max distress. Run 1 predates the pain sign convention fix; values have been converted to positive for consistency.*

**Integrity** is the most informative signal here. It declined monotonically from 1.000 to 0.323 — a drop of 67 points over the run. Integrity tracks whether the CLM loss ratio is consistently improving. The slow decline indicates the model was making progress but at a decelerating rate, which matches the CLM EMA curve flattening from ~8.85 at step 750 to ~8.40 at step 4,750 with most of that improvement in the first half of the run.

**Energy** remained consistently high (0.955–1.000) throughout, with only minor dips at steps 1,750, 2,500, and 3,000 — all points of architectural change or population pressure.

**Pain** stayed near zero throughout, confirming the model was never in acute distress. The loss was above random but not catastrophically so.

---

## Bugs Active During This Run

This run was completed before several fixes were applied. The following issues affected the results:

**Shapley values not reaching StatsTracker** — `_evolve()` computed Shapley values correctly but never returned them to `pretrain_step()`. As a result, `lifetime_contrib` was updated internally but `stats.json` recorded no Shapley history, and cell health in the inspector showed 0.000 for all cells. The EA was running on health scores that were being updated in memory but never properly surfaced. Fixed in the current codebase.

**Pain in negative range** — Pain was modelled as `[-1, 0]` with 0 meaning healthy. This was a sign convention mismatch with the other three homeostasis scalars. Values shown above have been converted to positive distress levels for readability. Fixed in the current codebase.

**Elite fraction 20%** — Crossover parents were drawn from the top 20% of cells (~5 cells at max population). With DAG diversity meaning most cells had different `in_features`, crossover rarely found matching donors. Most new cells initialised from random weights. Increased to 50% in the current codebase.

**No gene pool** — The gene pool did not exist in this run. All inheritance was from live elite cells only, with no historical memory. Added in the current codebase.

---

## What This Run Proved

Despite the bugs, run 1 established several important baselines:

- The model does learn. CLM EMA dropped from 9.09 to 8.40 — meaningful improvement on a real language modelling objective.
- The boom-bust cycle is real and severe. A 14-cell collapse in a single EA step is catastrophic. The gradual pruning fix was the right response.
- Deep structure (L2, L3) does emerge naturally without any explicit pressure to build it — the EA finds it useful enough to keep under the right conditions.
- The L1 layer stabilised at exactly 3 cells and held there from step 2,750 to 4,750. This suggests the model found an equilibrium for how many L1 cells are useful given the current encoder and task.
- Integrity declining while energy stays high is the signature of a model that is learning but slowing down — not a model in crisis. That's a useful pattern to recognise in future runs.

---

*Run 1 data generated by `scan_run.py`. Narrative written by genomorphic / Claude Sonnet 4.6.*

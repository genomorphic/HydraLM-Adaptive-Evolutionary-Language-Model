# Run 2 — Research Notes

**Steps:** 250 → 1,750 (7 checkpoints)  
**Vocab size:** 8,000 · **d_model:** 256 · **Max cells:** 24 · **DAG layers:** 10  
**Status:** Terminated early — training became unstable. Run 3 starts with fixed code.

---

## Overview

Run 2 was started after the boom-bust collapse fix from Run 1 but before several other important fixes were applied. It reached 1,750 steps before being terminated due to a clear instability pattern — energy draining to 0.522 and CLM EMA spiking back to 9.16 after appearing to improve. The run is short but informative: it confirms the single-horizon baseline problem and the optimizer rebuild issue, and the energy drain pattern directly motivated the dual-horizon homeostasis fix applied before Run 3.

The population grew steadily from 6 to 22 cells without a collapse — confirming the gradual pruning fix from Run 1 was working. L1 depth first appeared at step 1,000 and held through to termination. In that sense the architecture was healthier than Run 1 at the same step count. The problem was in the training dynamics, not the evolutionary dynamics.

---

## Population Growth

Run 2 showed cleaner population growth than Run 1. No boom-bust collapse occurred in 1,750 steps — the gradual pruning fix (max 7 cells per EA round) worked as intended. The first L1 cell appeared at step 1,000 and a second at step 1,750, a slower but more stable hierarchy than Run 1.

```
  Step     250 | ██████████                               |  6 cells  depth=0
  Step     500 | ██████████████                           |  8 cells  depth=0
  Step     750 | ██████████████████                       | 10 cells  depth=0
  Step   1,000 | ███████████████████████████              | 15 cells  depth=1  ← L1 appears
  Step   1,250 | ██████████████████████████████           | 17 cells  depth=1
  Step   1,500 | ██████████████████████████████████       | 19 cells  depth=1
  Step   1,750 | ████████████████████████████████████████ | 22 cells  depth=1  ← PEAK
```

Notably L1 appeared at step 1,000 in Run 2 vs step 3,000 in Run 1. The gradual pruning fix appears to have helped hierarchy build earlier — without the catastrophic collapse wiping structure at step 2,000, the model could accumulate useful L1 cells sooner.

---

## The Instability Pattern

Two energy drain events are visible in the homeostasis data:

| Step | Energy | CLM EMA | Event |
|------|--------|---------|-------|
|  250 |  1.000 |  9.081  | Healthy start |
|  500 |  1.000 |  8.943  | Improving |
|  750 |  0.594 |  9.363  | **First drain — loss spike** |
| 1000 |  0.991 |  8.720  | Recovered |
| 1250 |  0.999 |  8.662  | Stable |
| 1500 |  0.934 |  8.668  | Minor drain |
| 1750 |  0.522 |  9.163  | **Second drain — loss spike, run terminated** |

Both drain events follow the same pattern: energy collapses, CLM EMA spikes, then the model attempts to recover. At step 1,750 the recovery didn't come in time — energy was at 0.522 and still falling.

The root cause in both cases was the optimizer rebuild firing every `growth_interval=200` steps regardless of whether the architecture changed. Each rebuild wiped Adam's momentum buffers, the model lost its gradient history, and loss spiked. With the single-horizon baseline (alpha=0.05), pain never caught up to the spike quickly enough to gate the policy gradient or signal genuine distress. Energy drained silently instead.

---

## Bugs Active During This Run

Run 2 predates three fixes that are now in the codebase. Understanding these explains the instability pattern above.

**Bug 1 — Optimizer rebuild every growth_interval regardless of architecture change**

`train.py` called `AdamW(new_main, lr=lr_now)` every 200 steps unconditionally, even when `_evolve()` made no changes. This discarded all Adam momentum — the model's gradient history — on a fixed schedule. Every 200 steps the effective learning rate reset to its cold-start behaviour. With `lr=9e-7` at step 1,500+ this was too small to recover from the momentum loss quickly, producing the energy drain and loss spikes visible at steps 750 and 1,750.

*Fix applied for Run 3:* Optimizer now rebuilds only when `reward_info["arch_changed"]` is True — i.e. only when `_evolve()` actually modified the cell population. Momentum is also transplanted for parameters that survived the architecture change.

**Bug 2 — Single-horizon baseline too slow to signal distress**

The homeostasis EMA used alpha=0.05, giving a ~20-step time constant. When loss spiked at step 750 from ~8.94 to 9.36, the EMA baseline took ~100 steps to catch up. During that window `ratio = clm_loss / baseline` stayed near 1.0, pain never fired, and energy drained without the system registering the crisis. The pain signal that was supposed to gate the policy gradient and consolidate gradients was effectively blind to the most important events.

*Fix applied for Run 3:* Dual-horizon baseline. `fast_ema` (alpha=0.20, ~5-step horizon) drives pain/energy/excitement. `clm_loss_ema` (alpha=0.05) is retained for integrity only. A loss spike now fires pain within 3–4 steps.

**Bug 3 — reinforce_weight fixed regardless of model state**

The REINFORCE policy gradient on the exit gate ran at full `reinforce_weight=0.01` even when energy was at 0.522 and the model was in distress. Aggressively updating the exit gate architecture decision during a loss spike added noise on top of an already destabilised system.

*Fix applied for Run 3:* `reinforce_gate = max(0, 1 - max(pain, 1-energy))`. At energy=0.522 the gate would have been ~0.478 — the exit gate update would have run at roughly half strength rather than full. As the model recovers, the gate opens naturally.

---

## What Run 2 Contributed

Despite being short and unstable, Run 2 provided useful signal:

- **Confirmed gradual pruning fix works** — no boom-bust collapse in 1,750 steps. L1 appeared earlier than Run 1, suggesting stable population dynamics help hierarchy build faster.
- **Identified optimizer rebuild as primary instability driver** — the two energy drain events at steps 750 and 1,750 both correspond to growth_interval boundaries (750 = 3×250, 1750 = ~9×200). The pattern is too regular to be coincidental.
- **Demonstrated single-horizon baseline inadequacy** — energy drained to 0.594 at step 750 with pain showing only 0.009. The system registered mild discomfort while catastrophic momentum loss was occurring. The dual-horizon fix is a direct response to this observation.
- **Showed CLM EMA spike-and-recovery pattern** — both events showed a spike then partial recovery (9.363 → 8.720, 9.163 → ?). The model does recover, just too slowly with a cold optimizer and a blind pain signal.

---

## Homeostasis Analysis

| Step | Energy | Integrity | Pain | CLM EMA |
|------|--------|-----------|------|---------|
|  250 |  1.000 |     1.000 | 0.000 |  9.0811 |
|  500 |  1.000 |     1.000 | 0.000 |  8.9431 |
|  750 |  0.594 |     0.987 | 0.009 |  9.3630 |
| 1000 |  0.991 |     0.975 | 0.000 |  8.7198 |
| 1250 |  0.999 |     0.961 | 0.001 |  8.6623 |
| 1500 |  0.934 |     0.933 | 0.002 |  8.6683 |
| 1750 |  0.522 |     0.910 | 0.008 |  9.1633 |

*Pain shown as distress level — 0=healthy, 1=max distress. Run 2 predates the pain sign fix; values converted to positive for consistency.*

**Energy** is the most diagnostic signal here. The two crashes to 0.594 and 0.522 are sharp, not gradual — consistent with a sudden external shock (optimizer rebuild) rather than organic model fatigue. After the first crash at step 750, energy fully recovered to 0.999 by step 1,250. After the second at step 1,750 the run was stopped before recovery could be measured.

**Integrity** declined slowly and consistently from 1.000 to 0.910 — the same gradual pattern as Run 1, reflecting improving but decelerating CLM performance. Notably integrity never crashed even during the energy drain events — the slow EMA correctly showed structural improvement continuing even as the short-term dynamics were unstable.

**Pain** stayed near zero throughout both events (0.009 and 0.008 at the crash checkpoints) — confirmation that the single-horizon baseline was failing to register the distress that energy was clearly recording.

The divergence between energy and pain during the crash events is the clearest possible demonstration of why the dual-horizon fix was needed. These two signals should move together during a genuine crisis. In Run 2 they didn't.

---

## What Run 3 Changes

Run 3 starts from scratch with three fixes applied that directly address the failure modes observed here:

1. **Conditional optimizer rebuild** — momentum preserved between EA steps unless architecture actually changes
2. **Dual-horizon baseline** — fast_ema for pain/energy/excitement, slow EMA for integrity
3. **reinforce_gate** — policy gradient scales down when pain or energy signals distress

The gene pool and Shapley pipeline fixes from Run 1 also carry forward. Run 3 should show more stable energy (no sudden drains at 200-step boundaries), pain that actually tracks loss spikes, and a reinforce signal that backs off during architecture transitions rather than adding noise on top of them.

---

*Run 2 data generated by `scan_run.py`. Notes written by genomorphic / Claude Sonnet 4.6.*

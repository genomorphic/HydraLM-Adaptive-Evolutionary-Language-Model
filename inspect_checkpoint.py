"""
HydraLM Checkpoint Inspector
Reads a DNA checkpoint and prints a comprehensive report.
Some metrics (Shapley trends, loop distribution etc.) require
a stats.json file written by the instrumented training loop.
"""

import json, sys, math, os
from pathlib import Path
from collections import defaultdict

def load_json(path):
    with open(path) as f:
        return json.load(f)

def bar(val, min_val=0.0, max_val=1.0, width=20, fill="█", empty="░"):
    frac  = max(0.0, min(1.0, (val - min_val) / max(max_val - min_val, 1e-8)))
    n     = int(frac * width)
    return fill * n + empty * (width - n)

def fmt(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def subsection(title):
    print(f"\n  ── {title} ──")

# ── CLI ──────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <path/to/ckpt.json> [stats.json]")
    sys.exit(1)

ckpt_path  = Path(sys.argv[1])
stats_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ckpt_path.parent / "stats.json"

if not ckpt_path.exists():
    print(f"Checkpoint not found: {ckpt_path}"); sys.exit(1)

dna   = load_json(ckpt_path)
cfg   = dna.get("config", {})
cells = dna.get("cell_dna", [])

# Load companion .pt weights for buffer values (health, age, homeostasis)
import torch
pt_path = ckpt_path.with_suffix(".pt")
state_dict = {}
if pt_path.exists():
    state_dict = torch.load(str(pt_path), map_location="cpu")
    print(f"[Weights file loaded: {pt_path.name}]")
else:
    print(f"[No .pt file found at {pt_path.name} — health/age/homeostasis from .pt unavailable]")

def _buf(key, default=None):
    """Pull a scalar tensor from state_dict, return as Python float."""
    t = state_dict.get(key)
    if t is None:
        return default
    return t.item() if hasattr(t, "item") else float(t)

# Load stats if available
stats = {}
if stats_path.exists():
    stats = load_json(stats_path)
    print(f"[Stats file loaded: {stats_path}]")
else:
    print(f"[No stats file found at {stats_path} — some metrics unavailable]")

# ── 1. OVERVIEW ──────────────────────────────────────────────
section("OVERVIEW")
print(f"  DNA version     : {dna.get('dna_version', '?')}")
print(f"  Vocab size      : {cfg.get('vocab_size', '?'):,}")
print(f"  Encoder dim     : {cfg.get('encoder_dim', '?')}")
print(f"  Max hidden dim  : {cfg.get('max_hidden_dim', '?')}")
print(f"  Cell width range: [{cfg.get('min_width','?')}, {cfg.get('max_width','?')}]")
print(f"  DAG layers      : {cfg.get('n_dag_layers', '?')}")
print(f"  Max cells       : {cfg.get('max_cells', '?')}")
print(f"  Max connections : {cfg.get('max_connections', '?')}")
print(f"  Max loops       : {cfg.get('max_loops', '?')}")
print(f"  Replay ratio    : {cfg.get('replay_ratio', '?')}")
replay_seqs = dna.get("replay_seqs", [])
print(f"  Replay buf size : {len(replay_seqs):,} seqs")
print(f"  Replay seen     : {dna.get('replay_seen', 0):,}")

# ── 2. HOMEOSTASIS ───────────────────────────────────────────
section("HOMEOSTASIS")
homeo = stats.get("homeostasis_current", {})
# Fall back to .pt buffers when stats.json homeostasis is absent
_homeo_pt = {
    "energy":       _buf("homeostasis.energy"),
    "integrity":    _buf("homeostasis.integrity"),
    "excitement":   _buf("homeostasis.excitement"),
    "pain":         _buf("homeostasis.pain"),
    "clm_loss_ema": _buf("homeostasis.clm_loss_ema"),
}
for key, label in [
    ("energy",     "Energy    "),
    ("integrity",  "Integrity "),
    ("excitement", "Excitement"),
    ("pain",       "Pain      "),
]:
    val = homeo.get(key) if homeo.get(key) is not None else _homeo_pt.get(key)
    if val is not None:
        if key == "pain":
            # Pain is [-1, 0]: 0 = healthy, -1 = max distress.
            # Bar shows distress: full bar = worst pain, empty = no pain.
            distress  = abs(val)
            pain_bar  = bar(distress, 0.0, 1.0)
            label_str = ("no pain" if distress < 0.01 else
                         "mild"    if distress < 0.3  else
                         "moderate" if distress < 0.7 else "severe")
            print(f"  {label} : {fmt(val)} {pain_bar} [{label_str}]")
        else:
            print(f"  {label} : {fmt(val)} {bar(val, 0.0, 1.0)}")
    else:
        print(f"  {label} : N/A")

clm_ema = homeo.get("clm_loss_ema") or _homeo_pt.get("clm_loss_ema")
if clm_ema:
    print(f"  CLM loss EMA   : {fmt(clm_ema)}")

# ── 3. CELL POPULATION ───────────────────────────────────────
section(f"CELL POPULATION  ({len(cells)} cells)")

# Layer distribution
layer_counts = defaultdict(int)
for c in cells:
    layer_counts[c["layer_idx"]] += 1

subsection("Layer distribution")
for layer in sorted(layer_counts):
    count = layer_counts[layer]
    print(f"  L{layer:2d} : {'●' * count}  ({count} cells)")

subsection("Cell details")
print(f"  {'ID':>3} {'L':>2} {'W_in':>6} {'W_out':>6} {'Srcs':>5} {'Health':>8} {'Age':>7} {'Tenure':>8} {'Frz':>4}")
print(f"  {'-'*3} {'-'*2} {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*7} {'-'*8} {'-'*4}")

cell_stats = stats.get("cell_stats", {})
for i, c in enumerate(cells):
    cs      = cell_stats.get(str(i), {})
    # Prefer stats.json values; fall back to .pt buffers
    health  = cs.get("health")
    if health is None:
        health = _buf(f"cells.{i}.health", "?")
    age     = cs.get("age")
    if age is None:
        _age = _buf(f"cells.{i}.age")
        age  = int(_age) if _age is not None else "?"
    tenure  = fmt(c.get("lifetime_contrib", 0.0), 3)
    frozen  = "yes" if c.get("frozen") else "no"
    n_src   = len(c.get("input_sources", []))
    w_in    = c.get("in_features", "?")
    w_out   = c.get("out_features", "?")
    h_bar   = bar(float(health), 0, 1, 10) if isinstance(health, (int, float)) else "?" * 10
    print(f"  {i:>3} {c['layer_idx']:>2} {w_in:>6} {w_out:>6} {n_src:>5} "
          f"  {fmt(health,3) if isinstance(health,(int,float)) else '?':>6} {h_bar} "
          f"{str(age):>7} {tenure:>8} {frozen:>4}")

subsection("DAG wiring")
for i, c in enumerate(cells):
    srcs = c.get("input_sources", [])
    if srcs:
        print(f"  Cell {i:>2} (L{c['layer_idx']}) ← {srcs}")
    else:
        print(f"  Cell {i:>2} (L{c['layer_idx']}) ← [root]")

# ── 4. SHAPLEY / HEALTH METRICS ──────────────────────────────
section("SHAPLEY & HEALTH METRICS")
sv_history = stats.get("shapley_history", {})
if sv_history:
    subsection("Shapley values (last recorded)")
    for i, c in enumerate(cells):
        sv_list = sv_history.get(str(i), [])
        if sv_list:
            last = sv_list[-1]
            trend = sv_list[-1] - sv_list[0] if len(sv_list) > 1 else 0.0
            trend_sym = "↑" if trend > 0.01 else ("↓" if trend < -0.01 else "→")
            print(f"  Cell {i:>2}: sv={fmt(last,3)} {bar(last,0,1,15)} trend={fmt(trend,3)}{trend_sym}")
    subsection("Activation magnitude")
    act_mag = stats.get("activation_magnitude", {})
    for i in range(len(cells)):
        mag = act_mag.get(str(i))
        if mag is not None:
            print(f"  Cell {i:>2}: mean_act={fmt(mag,4)}")
    subsection("Usage frequency")
    usage = stats.get("cell_usage_frequency", {})
    for i in range(len(cells)):
        freq = usage.get(str(i))
        if freq is not None:
            print(f"  Cell {i:>2}: usage={fmt(freq,4)} {bar(freq,0,1,15)}")
else:
    print("  Not available — requires stats.json from instrumented trainer")

# ── 5. LOOP / EXIT GATE ──────────────────────────────────────
section("LOOP & EXIT GATE")
loop_stats = stats.get("loop_stats", {})
if loop_stats:
    avg = loop_stats.get("avg_loops")
    dist = loop_stats.get("loop_distribution", {})
    exit_ent = loop_stats.get("exit_entropy")
    exit_by_loop = loop_stats.get("exit_probability_by_loop", {})
    loss_per_loop = loop_stats.get("loss_improvement_per_loop", {})
    tax = loop_stats.get("compute_tax_paid")

    if avg: print(f"  Avg loops/token     : {fmt(avg,3)}")
    if exit_ent: print(f"  Exit entropy        : {fmt(exit_ent,4)}")
    if tax: print(f"  Compute tax paid    : {fmt(tax,4)}")

    if dist:
        subsection("Loop distribution")
        total = sum(dist.values())
        for k in sorted(dist, key=int):
            v = dist[k]
            pct = v / max(total, 1)
            print(f"  Loop {k}: {pct*100:5.1f}% {bar(pct,0,1,20)}")

    if exit_by_loop:
        subsection("Exit probability by loop")
        for k in sorted(exit_by_loop, key=int):
            v = exit_by_loop[k]
            print(f"  Loop {k}: p_exit={fmt(v,3)} {bar(v,0,1,20)}")

    if loss_per_loop:
        subsection("Loss improvement per loop")
        for k in sorted(loss_per_loop, key=int):
            v = loss_per_loop[k]
            sym = "↑" if v > 0 else "↓"
            print(f"  Loop {k}: Δloss={fmt(v,4)} {sym}")
else:
    print("  Not available — requires stats.json from instrumented trainer")

# ── 6. LOSS COMPONENTS ───────────────────────────────────────
section("LOSS COMPONENTS (recent averages)")
loss_stats = stats.get("loss_stats", {})
if loss_stats:
    for key, label in [
        ("mlm_loss",    "MLM loss      "),
        ("clm_loss",    "CLM loss      "),
        ("cell_loss",   "Cell loss     "),
        ("exit_entropy","Exit entropy  "),
        ("compute_tax", "Compute tax   "),
        ("reinforce",   "Policy gradient"),
        ("total_loss",  "Total loss    "),
    ]:
        val = loss_stats.get(key)
        if val is not None:
            print(f"  {label}: {fmt(val)}")
else:
    print("  Not available — requires stats.json from instrumented trainer")

# ── 7. EA / EVOLUTION ────────────────────────────────────────
section("EVOLUTIONARY DYNAMICS")
ea_stats = stats.get("ea_stats", {})
if ea_stats:
    print(f"  Birth rate          : {fmt(ea_stats.get('birth_rate', 'N/A'))}")
    print(f"  Death rate          : {fmt(ea_stats.get('death_rate', 'N/A'))}")
    print(f"  Avg cell age        : {fmt(ea_stats.get('avg_cell_age', 'N/A'))}")
    print(f"  Lineage depth       : {fmt(ea_stats.get('lineage_depth', 'N/A'))}")
    catastrophes = ea_stats.get("catastrophe_events", [])
    mutations    = ea_stats.get("mutation_types", {})
    print(f"  Catastrophe events  : {len(catastrophes)}")
    if mutations:
        print(f"  Mutation types:")
        for mt, count in mutations.items():
            print(f"    {mt}: {count}")
else:
    print("  Not available — requires stats.json from instrumented trainer")
# Always show from DNA
avg_age = sum(c.get("lifetime_contrib", 0) for c in cells) / max(len(cells), 1)
print(f"  Avg lifetime_contrib: {avg_age:.4f}")
frozen_count = sum(1 for c in cells if c.get("frozen"))
print(f"  Frozen cells        : {frozen_count}/{len(cells)}")

# ── 8. REPLAY BUFFER ─────────────────────────────────────────
section("REPLAY BUFFER")
print(f"  Buffer size         : {len(replay_seqs):,} sequences")
if replay_seqs:
    lengths = [len(s) for s in replay_seqs]
    print(f"  Seq length  min/avg/max: "
          f"{min(lengths)} / {sum(lengths)/len(lengths):.1f} / {max(lengths)}")
replay_stats = stats.get("replay_stats", {})
if replay_stats:
    print(f"  Replay frequency    : {fmt(replay_stats.get('replay_frequency','N/A'))}")
    print(f"  Avg reward          : {fmt(replay_stats.get('avg_reward','N/A'))}")
    print(f"  Trajectory age avg  : {fmt(replay_stats.get('trajectory_age_avg','N/A'))}")
    rd = replay_stats.get("reward_distribution", {})
    if rd:
        subsection("Reward distribution")
        for bucket, count in sorted(rd.items()):
            print(f"  {bucket}: {count}")
else:
    print("  Extended stats not available — requires stats.json")

# ── 9. GRADIENTS ─────────────────────────────────────────────
section("GRADIENT STATS")
grad_stats = stats.get("gradient_stats", {})
if grad_stats:
    print(f"  Global grad norm    : {fmt(grad_stats.get('grad_norm','N/A'))}")
    print(f"  Grad variance       : {fmt(grad_stats.get('grad_variance','N/A'))}")
    per_layer = grad_stats.get("per_layer", {})
    if per_layer:
        subsection("Per-layer gradient magnitude")
        for layer, mag in sorted(per_layer.items()):
            print(f"  {layer:30s}: {fmt(mag,6)}")
else:
    print("  Not available — requires stats.json from instrumented trainer")

# ── 10. GENE POOL ────────────────────────────────────────────
section("GENE POOL")
gp_path = ckpt_path.with_name(ckpt_path.stem + "_genepool.pt")
if gp_path.exists():
    try:
        gp_data  = torch.load(str(gp_path), map_location="cpu")
        gp_pool  = gp_data.get("pool", [])
        gp_ptr   = gp_data.get("ptr",  0)
        capacity = 200

        print(f"  Snapshots       : {len(gp_pool)} / {capacity} "
              f"({'full' if len(gp_pool) >= capacity else 'filling'})")
        print(f"  Total donations : {gp_ptr}")

        if gp_pool:
            # Source breakdown
            sources = {}
            for s in gp_pool:
                src = s.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            src_str = "  ".join(f"{k}={v}" for k, v in sorted(sources.items()))
            print(f"  By source       : {src_str}")

            # Layer distribution
            layer_counts = {}
            for s in gp_pool:
                l = s.get("layer", "?")
                layer_counts[l] = layer_counts.get(l, 0) + 1
            subsection("Layer distribution")
            for layer in sorted(k for k in layer_counts if isinstance(k, int)):
                count = layer_counts[layer]
                pct   = count / len(gp_pool)
                print(f"  L{layer:2d} : {'●' * count}  ({count} snapshots, {pct*100:.0f}%)")

            # Shape diversity
            subsection("Shape diversity (in_features × out_features)")
            shapes = {}
            for s in gp_pool:
                key = (s.get("in", "?"), s.get("out", "?"))
                shapes[key] = shapes.get(key, 0) + 1
            # Show unique shapes sorted by frequency
            for (in_f, out_f), count in sorted(shapes.items(),
                                               key=lambda x: -x[1])[:10]:
                print(f"  {in_f:>6} × {out_f:<6}  ×{count}")
            if len(shapes) > 10:
                print(f"  ... and {len(shapes)-10} more unique shapes")

            # Weight magnitude stats across pool
            subsection("Weight magnitude")
            means = [s["w"].abs().mean().item() for s in gp_pool if "w" in s]
            if means:
                print(f"  Mean |w| across pool: "
                      f"min={min(means):.4f}  "
                      f"avg={sum(means)/len(means):.4f}  "
                      f"max={max(means):.4f}")
    except Exception as e:
        print(f"  Could not load gene pool: {e}")
else:
    print(f"  No gene pool file found ({gp_path.name})")
    print(f"  (Gene pool saves alongside DNA from hydra_lm.py >= gene pool update)")

print(f"\n{'='*60}")
print(f"  Inspection complete")
print(f"{'='*60}\n")

# Global Gene Pool

This folder contains the latest consolidated gene pool built from community contributions.

The global pool gives your model a head start — instead of every new cell initialising from random weights, it inherits from the best cells produced across all community runs.

---

## How to use it

**1. Download `global_genepool.pt`**

**2. Rename it to match your checkpoint**

The file name must match your checkpoint's base name with `_genepool.pt` appended:

```
ckpt_step00000000_genepool.pt
```

If you are starting a brand new run (step 0), use:
```
ckpt_step00000000_genepool.pt
```

If you are resuming from an existing checkpoint, match that step number:
```
ckpt_step00005000_genepool.pt
```

**3. Drop it in your run folder**

Place it alongside your other checkpoint files:
```
runs/run1/
├── ckpt_step00005000.json
├── ckpt_step00005000.pt
├── ckpt_step00005000_genepool.pt   ← here
├── ckpt_opt_00005000.pt
└── ckpt_latest.txt
```

`HydraLM.load_dna()` will find and load it automatically on the next resume.

---

## What is in the pool

The global pool contains up to 200 weight snapshots selected from community contributions. Each snapshot is a cell that either earned a freeze (survived long enough to prove consistent high contribution) or was elite at the time of pruning (useful but evicted by population pressure).

Snapshots are validated for weight health and selected to maximise diversity across DAG layers and weight shapes — so the pool contains structure from shallow and deep cells across different architectures.

The `source` field on each snapshot tells you how it was donated: `freeze` or `prune`.

---

## Current pool

| Built | Snapshots | Contributors | Layer distribution |
|-------|-----------|--------------|-------------------|
| —     | —         | —            | —                 |

*(Updated when a new global pool is published)*

---

## Want to contribute?

See the `contribute/` folder.

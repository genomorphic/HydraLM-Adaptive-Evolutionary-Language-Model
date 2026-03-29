# Contribute Your Gene Pool

Thank you for training HydraLM. If your run produced healthy cells, sharing your gene pool helps every future run start with better inherited weights.

---

## What you are sharing

Your `*_genepool.pt` file contains weight snapshots from your best-performing cells — cells that either earned a freeze (proven long-term utility) or were elite at the time of pruning. It does not contain your full model weights, your training data, or any personally identifiable information.

---

## How to contribute

**1. Find your gene pool file**

It lives in your run folder alongside your checkpoints:
```
runs/run1/ckpt_step00005000_genepool.pt
```

Pick the checkpoint where your model was performing best — highest health scores, deepest DAG, lowest perplexity.

**2. Rename it**

Add your GitHub username as a prefix to avoid collisions:
```
[username]ckpt_step00005000_genepool.pt
```

For example:
```
alice ckpt_step00005000_genepool.pt
```

**3. Upload it here**

Open a pull request adding your renamed file to this `contribute/` folder. One file per contributor is enough — if you have multiple runs, pick your best.

---

## What happens next

We periodically run `build_pool.py` across all contributions, validate each snapshot, select the best by quality and diversity, and publish a consolidated `global_genepool.pt` in the `global_pool/` folder.

Your snapshots are scored and selected based on weight health — near-zero or unstable weights are automatically excluded. Layer diversity is preserved so the global pool contains structure from all DAG depths, not just L0 cells.

---

## Notes

- There is no minimum run length required, but longer runs with lower perplexity produce better donations
- Contributions from any `d_model` size are welcome — the gene pool's shape projection handles architecture mismatches automatically
- You can contribute again if a later run produces significantly better cells

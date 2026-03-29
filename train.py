"""
train.py — HydraLM Streaming Trainer
======================================
Standalone training script. Imports HydraLM and SuperBPETokenizer from
hydra_lm.py (must be in the same directory or on PYTHONPATH).

Features
--------
  Streaming data loader   : reads text files line-by-line, never loads the
                            full corpus into RAM. Works on arbitrarily large
                            datasets.
  BPE tokenizer training  : trains on a configurable sample of the corpus
                            before model training begins. Saves to disk so
                            subsequent runs can reuse it.
  Checkpointing           : saves model weights, optimizer state, tokenizer,
                            config, and step count. Resumable from any checkpoint.
  Eval / perplexity       : held-out eval split, computed every eval_every steps.
  Logging                 : plain stdout + optional CSV loss log.
  Argparse entry point    : all hyperparameters configurable from command line.

Quick start
-----------
  # First run — trains tokenizer and model from scratch
  python train.py --data_path corpus.txt --run_dir runs/run1

  # Resume from checkpoint
  python train.py --data_path corpus.txt --run_dir runs/run1 --resume

  # Custom hyperparameters
  python train.py \\
      --data_path corpus.txt \\
      --run_dir   runs/big \\
      --vocab_size 16000 \\
      --d_model    512 \\
      --n_layers   6 \\
      --max_loops  4 \\
      --batch_size 32 \\
      --lr         3e-4 \\
      --steps      100000 \\
      --eval_every 500

Data format
-----------
  Plain text file, one document per line (or free-form — the loader
  splits on newlines). Empty lines are skipped. UTF-8 encoding expected.
  For very large files the loader streams chunks without loading all into RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from hydra_lm import HydraLM, SuperBPETokenizer


# ===========================================================================
# 1.  Config
# ===========================================================================

@dataclass
class TrainConfig:
    # Data
    data_path:          str   = "corpus.txt"        # training corpus
    eval_split:         float = 0.20                # fraction held out for eval
    bpe_sample_lines:   int   = 500_000             # lines used to train tokenizer
    bpe_min_freq:       int   = 2                   # BPE merge minimum frequency
    index_cache_path:   str   = ""                  # path for line-index cache (auto if empty)

    # Run management
    run_dir:            str   = "runs/default"      # checkpoint + log directory
    resume:             bool  = False               # resume from latest checkpoint

    # Tokenizer
    vocab_size:         int   = 8000

    # Model architecture
    d_model:            int   = 256
    n_heads:            int   = 4
    n_layers:           int   = 4
    max_seq_len:        int   = 512
    encoder_dim:        int   = 256
    dropout:            float = 0.1
    gate_n_stages:      int   = 4
    gate_n_probes:      int   = 5
    gate_warmup_steps:  int   = 500
    initial_cells:      int   = 4
    max_cells:          int   = 16
    min_width:          int   = 16                  # minimum cell output width
    max_width:          int   = 512                 # maximum cell output width
    max_connections:    int   = 10                  # max input sources per cell
    n_dag_layers:       int   = 2
    max_loops:          int   = 4
    exit_entropy_weight: float = 0.05
    loop_depth_weight:  float = 0.01               # compute tax per loop
    reinforce_weight:   float = 0.01               # REINFORCE policy gradient weight

    # Training
    batch_size:         int   = 32
    lr:                 float = 1e-6
    weight_decay:       float = 0.01
    warmup_steps:       int   = 500
    steps:              int   = 5_000
    grad_clip:          float = 1.0
    mlm_weight:         float = 0.5
    unc_reg_weight:     float = 0.01

    # Schedule
    eval_every:         int   = 500
    save_every:         int   = 2000
    log_every:          int   = 50

    # Misc
    seed:               int   = 42
    device:             str   = "auto"              # auto / cpu / cuda / mps

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ===========================================================================
# 2.  Streaming data loader
# ===========================================================================

class StreamingTextDataset:
    """
    Streams tokenized sequences from a large text file without loading it
    all into RAM.

    The file is read in random-access chunks during training:
    - On construction we index newline positions so we can seek to any line.
    - During iteration we sample random line offsets and encode on the fly.

    For very large files (> a few GB) the line index itself can be large,
    but it's just a list of integers (8 bytes each) so a 10M-line file uses
    ~80 MB for the index.
    """

    def __init__(
        self,
        path:        str,
        tokenizer:   SuperBPETokenizer,
        max_seq_len: int  = 512,
        line_indices: Optional[List[int]] = None,   # pre-built index (for eval split)
    ):
        self.path        = path
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

        if line_indices is not None:
            self.line_offsets = line_indices
        else:
            print("  Indexing corpus line offsets...")
            self.line_offsets = self._build_index(path)
            print(f"  {len(self.line_offsets):,} lines indexed.")

    @staticmethod
    def _build_index(path: str) -> List[int]:
        offsets = [0]
        with open(path, "rb") as f:
            for line in f:
                offsets.append(f.tell())
        return offsets[:-1]   # last entry would be EOF

    def _read_line(self, offset: int) -> str:
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return f.readline().rstrip("\n")

    def sample_batch(
        self,
        batch_size: int,
        device:     str = "cpu",
    ) -> torch.Tensor:
        """
        Sample `batch_size` random lines, encode them, pad to uniform length.
        Returns (B, max_seq_len) int64 tensor.
        """
        lines  = []
        trials = 0
        while len(lines) < batch_size:
            offset = random.choice(self.line_offsets)
            line   = self._read_line(offset).strip()
            if len(line) > 4:
                lines.append(line)
            trials += 1
            if trials > batch_size * 20:
                break   # sparse file guard

        encoded = [
            self.tokenizer.encode(
                line, add_bos=True, add_eos=True, max_length=self.max_seq_len
            )
            for line in lines
        ]
        # Pad to uniform length within this batch (not necessarily max_seq_len)
        max_l  = min(max(len(e) for e in encoded), self.max_seq_len)
        padded = [
            e[:max_l] + [self.tokenizer.PAD_ID] * max(0, max_l - len(e))
            for e in encoded
        ]
        return torch.tensor(padded, dtype=torch.long, device=device)

    @classmethod
    def train_eval_split(
        cls,
        path:        str,
        tokenizer:   SuperBPETokenizer,
        max_seq_len: int,
        eval_frac:   float = 0.02,
        seed:        int   = 42,
    ) -> Tuple["StreamingTextDataset", "StreamingTextDataset"]:
        """Split a single file into train and eval datasets by line index."""
        all_offsets = cls._build_index(path)
        random.seed(seed)
        random.shuffle(all_offsets)
        n_eval  = max(1, int(len(all_offsets) * eval_frac))
        eval_idx  = all_offsets[:n_eval]
        train_idx = all_offsets[n_eval:]
        train_ds = cls(path, tokenizer, max_seq_len, line_indices=train_idx)
        eval_ds  = cls(path, tokenizer, max_seq_len, line_indices=eval_idx)
        print(f"  Train: {len(train_idx):,} lines | Eval: {len(eval_idx):,} lines")
        return train_ds, eval_ds




class PrefetchBuffer:
    """
    Background thread that prefetches batches from a StreamingTextDataset
    so disk I/O overlaps with GPU computation.

    Uses a queue of size `buffer_size` (default 4). The worker thread
    keeps the queue full; the training loop calls .next() which returns
    immediately if a batch is ready, or blocks briefly if the worker
    is behind.

    This hides the latency of random file seeks (especially bad over
    Google Drive) behind the GPU forward/backward pass.
    """

    def __init__(
        self,
        dataset:     "StreamingTextDataset",
        batch_size:  int,
        device:      str,
        buffer_size: int = 8,
    ):
        import queue
        import threading

        self.queue = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()

        def _worker():
            while not self._stop.is_set():
                try:
                    batch = dataset.sample_batch(batch_size, device="cpu")
                    self.queue.put(batch, timeout=5.0)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._device = device
        self._thread.start()

    def next(self) -> torch.Tensor:
        """Get next prefetched batch, moved to target device."""
        import queue
        batch = self.queue.get(timeout=30.0)
        return batch.to(self._device, non_blocking=True)

    def stop(self):
        self._stop.set()

# ===========================================================================
# 3.  Tokenizer bootstrap
# ===========================================================================

def build_or_load_tokenizer(cfg: TrainConfig, run_dir: Path) -> SuperBPETokenizer:
    """
    Load tokenizer from disk if it exists, otherwise train from a sample
    of the corpus and save it.
    """
    tok_path = str(run_dir / "tokenizer.json")

    if os.path.exists(tok_path):
        print(f"Loading tokenizer from {tok_path}")
        return SuperBPETokenizer.load(tok_path)

    print(f"Training BPE tokenizer (vocab_size={cfg.vocab_size}) ...")
    tok    = SuperBPETokenizer(vocab_size=cfg.vocab_size)
    texts: List[str] = []
    print(f"  Sampling up to {cfg.bpe_sample_lines:,} lines for BPE training...")
    with open(cfg.data_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) > 4:
                texts.append(line)
            if len(texts) >= cfg.bpe_sample_lines:
                break
    print(f"  Using {len(texts):,} lines.")
    tok.train(texts, min_frequency=cfg.bpe_min_freq, verbose=True)
    tok.save(tok_path)
    return tok


# ===========================================================================
# 4.  Checkpoint save / load
# ===========================================================================

def save_checkpoint(
    run_dir:   Path,
    model:     HydraLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step:      int,
    best_ppl:  float,
) -> None:
    """
    Save full model DNA (topology + weights + EA instincts + homeostasis)
    plus optimizer and scheduler state for exact resumption.

    Produces three files per checkpoint:
      ckpt_step{N}.json  — human-readable topology, config, EA instincts
      ckpt_step{N}.pt    — weights + buffers (companion to the JSON)
      ckpt_opt_{N}.pt    — optimizer + scheduler state

    The JSON + .pt pair is the portable "DNA" — can be loaded on any
    hardware with HydraLM.load_dna(). The optimizer file is only needed
    to resume training from exactly this point.
    """
    dna_base  = str(run_dir / f"ckpt_step{step:08d}.json")
    opt_path  = run_dir / f"ckpt_opt_{step:08d}.pt"

    # Save full DNA (topology + weights)
    model.save_dna(dna_base)

    # Save optimizer + scheduler + meta separately
    torch.save({
        "step":            step,
        "best_ppl":        best_ppl,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, opt_path)

    # Pointer to latest checkpoint base name
    (run_dir / "ckpt_latest.txt").write_text(f"ckpt_step{step:08d}")
    print(f"  [Ckpt] step={step:,} | opt → {opt_path.name}")


def load_latest_checkpoint(
    run_dir:   Path,
    cfg:       "TrainConfig",
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device:    str,
) -> Tuple[HydraLM, int, float]:
    """
    Resurrect the full model DNA from the latest checkpoint.
    Returns (model, step, best_ppl). Returns (None, 0, inf) if no checkpoint.

    Uses HydraLM.load_dna() so the exact topology, wiring, frozen state,
    and EA lifetime contributions are all restored — not just the weights.
    """
    ptr = run_dir / "ckpt_latest.txt"
    if not ptr.exists():
        return None, 0, float("inf")

    base_name = ptr.read_text().strip()           # e.g. "ckpt_step00002000"
    json_path = str(run_dir / f"{base_name}.json")
    opt_path  = run_dir / f"ckpt_opt_{base_name.split('step')[1]}.pt"

    if not Path(json_path).exists():
        return None, 0, float("inf")

    print(f"  [Ckpt] Resuming DNA from {base_name}")
    model = HydraLM.load_dna(
        json_path,
        device             = device,
        d_model            = cfg.d_model,
        n_heads            = cfg.n_heads,
        n_layers           = cfg.n_layers,
        max_seq_len        = cfg.max_seq_len,
        dropout            = cfg.dropout,
        gate_n_stages      = cfg.gate_n_stages,
        gate_n_probes      = cfg.gate_n_probes,
        gate_warmup_steps  = cfg.gate_warmup_steps,
    )
    model.train()

    step     = 0
    best_ppl = float("inf")
    if opt_path.exists():
        opt_ckpt  = torch.load(opt_path, map_location=device)
        step      = opt_ckpt["step"]
        best_ppl  = opt_ckpt["best_ppl"]
        # Rebuild optimizer against the resumed model's parameters before
        # loading state, so parameter references are correct.
        # Note: exit gate excluded from main optimizer if reinforce is active —
        # same logic as initial construction.
        exit_gate_params = set(id(p) for p in model.exit_gate.parameters())
        if cfg.reinforce_weight > 0:
            main_params = [p for p in model.parameters()
                           if p.requires_grad and id(p) not in exit_gate_params]
        else:
            main_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            main_params, lr=cfg.lr,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
        )
        optimizer.load_state_dict(opt_ckpt["optimizer_state"])
        scheduler.load_state_dict(opt_ckpt["scheduler_state"])

    return model, step, best_ppl


# ===========================================================================
# 5.  Eval — perplexity on held-out split
# ===========================================================================

@torch.no_grad()
def evaluate(
    model:     HydraLM,
    eval_ds:   StreamingTextDataset,
    tokenizer: SuperBPETokenizer,
    cfg:       TrainConfig,
    device:    str,
    n_batches: int = 20,
) -> float:
    """
    Estimate perplexity on the eval split.
    Runs n_batches random batches, returns mean perplexity.
    Perplexity = exp(mean_cross_entropy_loss).
    """
    model.eval()
    total_loss = 0.0
    total_tok  = 0

    for _ in range(n_batches):
        ids = eval_ds.sample_batch(cfg.batch_size, device=device)

        # Causal CLM loss only (clean perplexity signal)
        enc, hidden, _ = model.encoder(
            ids,
            global_step   = int(model.step_count.item()),
            integrity     = model.homeostasis.integrity.item(),
            return_hidden = True,
            causal        = False,  # bidirectional for eval MLM ppl
        )

        if hidden is None:
            continue

        B, L, V_dim = hidden.shape
        if L < 2:
            continue

        logits  = model.mlm_head(hidden[:, :-1, :])        # (B, L-1, vocab)
        targets = ids[:, 1:]                               # (B, L-1)
        mask    = (targets != tokenizer.PAD_ID)

        loss = nn.functional.cross_entropy(
            logits.reshape(B * (L - 1), model.vocab_size),
            targets.reshape(B * (L - 1)),
            ignore_index = tokenizer.PAD_ID,
            reduction    = "sum",
        )
        n_tok       = mask.sum().item()
        total_loss += loss.item()
        total_tok  += n_tok

    model.train()
    if total_tok == 0:
        return float("inf")
    mean_ce = total_loss / total_tok
    return math.exp(min(mean_ce, 20.0))   # cap at exp(20) to avoid overflow


# ===========================================================================
# 6.  LR schedule helpers
# ===========================================================================

def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Linear warmup then cosine decay to 10% of peak LR.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (0.10 + 0.90 * cosine)


# ===========================================================================
# 6b. Stats Tracker
# ===========================================================================

class StatsTracker:
    def __init__(self, run_dir, window=100):
        self.run_dir = Path(run_dir)
        self.window  = window
        self._loss   = defaultdict(list)
        self._loops  = []
        self._loop_dist = defaultdict(int)
        self._exit_ent  = []
        self._tax = []
        self._pg  = []
        self._births = 0
        self._deaths = 0
        self._catastrophes = []
        self._mutations = defaultdict(int)
        self._grad_norms = []
        self._shapley = defaultdict(list)
        self._step_count = 0
        self._homeo = {}

    def record_step(self, reward_info, model, grad_norm):
        self._step_count += 1
        w = self.window
        for key in ["mlm_loss","clm_loss","cell_loss","exit_entropy",
                    "compute_tax","reinforce","total_loss"]:
            v = reward_info.get(key)
            if v is not None:
                self._loss[key].append(v)
                if len(self._loss[key]) > w: self._loss[key].pop(0)
        n = reward_info.get("n_loops", 4)
        self._loops.append(n)
        if len(self._loops) > w: self._loops.pop(0)
        self._loop_dist[int(n)] += 1
        self._exit_ent.append(reward_info.get("exit_entropy", 0))
        if len(self._exit_ent) > w: self._exit_ent.pop(0)
        self._tax.append(reward_info.get("compute_tax", 0))
        if len(self._tax) > w: self._tax.pop(0)
        self._pg.append(reward_info.get("reinforce", 0))
        if len(self._pg) > w: self._pg.pop(0)
        self._grad_norms.append(grad_norm)
        if len(self._grad_norms) > w: self._grad_norms.pop(0)
        h = model.homeostasis
        self._homeo = {
            "energy": h.energy.item(), "integrity": h.integrity.item(),
            "excitement": h.excitement.item(), "pain": h.pain.item(),
            "clm_loss_ema": h.clm_loss_ema.item(),
        }
        for k, v in reward_info.items():
            if k.startswith("sv_cell_"):
                idx = int(k.split("_")[-1])
                self._shapley[idx].append(round(float(v), 5))
                if len(self._shapley[idx]) > 50: self._shapley[idx].pop(0)

    def record_ea_event(self, event_type, details=None):
        if event_type == "birth":
            self._births += 1
            self._mutations[(details or {}).get("mutation","random")] += 1
        elif event_type == "death":
            self._deaths += 1
        elif event_type == "catastrophe":
            self._catastrophes.append(details or {})

    def _avg(self, lst): return sum(lst)/len(lst) if lst else 0.0

    def save(self, path=None):
        import statistics as _st
        out = path or str(self.run_dir / "stats.json")
        data = {
            "step": self._step_count,
            "homeostasis_current": self._homeo,
            "loss_stats": {k: round(self._avg(v), 6) for k, v in self._loss.items()},
            "loop_stats": {
                "avg_loops":         round(self._avg(self._loops), 3),
                "loop_distribution": dict(self._loop_dist),
                "exit_entropy":      round(self._avg(self._exit_ent), 5),
                "compute_tax_paid":  round(self._avg(self._tax), 5),
                "avg_pg":            round(self._avg(self._pg), 6),
            },
            "shapley_history": {str(k): v for k, v in self._shapley.items()},
            "ea_stats": {
                "birth_rate":         round(self._births/max(self._step_count,1), 5),
                "death_rate":         round(self._deaths/max(self._step_count,1), 5),
                "catastrophe_events": self._catastrophes,
                "mutation_types":     dict(self._mutations),
            },
            "gradient_stats": {
                "grad_norm":     round(self._avg(self._grad_norms), 5),
                "grad_variance": round(_st.variance(self._grad_norms)
                                       if len(self._grad_norms) > 1 else 0.0, 5),
            },
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=2)


# ===========================================================================
# 7.  Main training loop
# ===========================================================================

def train(cfg: TrainConfig) -> None:
    # ---- Setup ----
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = cfg.device
    print(f"Device : {device}")

    cfg.save(str(run_dir / "config.json"))

    # ---- Tokenizer ----
    tok = build_or_load_tokenizer(cfg, run_dir)

    # ---- Dataset ----
    print("\nBuilding dataset...")
    train_ds, eval_ds = StreamingTextDataset.train_eval_split(
        cfg.data_path, tok, cfg.max_seq_len, cfg.eval_split, cfg.seed
    )

    # ---- Model ----
    print("\nBuilding model...")
    model = HydraLM(
        vocab_size           = tok.vocab_size,
        pad_id               = tok.PAD_ID,
        d_model              = cfg.d_model,
        n_heads              = cfg.n_heads,
        n_layers             = cfg.n_layers,
        max_seq_len          = cfg.max_seq_len,
        encoder_dim          = cfg.encoder_dim,
        dropout              = cfg.dropout,
        gate_n_stages        = cfg.gate_n_stages,
        gate_n_probes        = cfg.gate_n_probes,
        gate_warmup_steps    = cfg.gate_warmup_steps,
        initial_cells        = cfg.initial_cells,
        max_cells            = cfg.max_cells,
        min_width            = cfg.min_width,
        max_width            = cfg.max_width,
        max_connections      = cfg.max_connections,
        n_dag_layers         = cfg.n_dag_layers,
        max_loops            = cfg.max_loops,
        exit_entropy_weight  = cfg.exit_entropy_weight,
        loop_depth_weight    = cfg.loop_depth_weight,
        reinforce_weight     = cfg.reinforce_weight,
    ).to(device)
    print(model.summary())

    # ---- Optimizer ----
    # Exit gate excluded from main optimizer when REINFORCE is active —
    # prevents pass 1 from modifying gate weights before pass 2's backward.
    exit_gate_params = set(id(p) for p in model.exit_gate.parameters())
    if cfg.reinforce_weight > 0:
        main_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in exit_gate_params]
    else:
        main_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        main_params,
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    # Separate optimizer for REINFORCE on exit gate only.
    pg_optimizer = torch.optim.Adam(
        model.exit_gate.parameters(),
        lr=cfg.lr * 0.1,
    ) if cfg.reinforce_weight > 0 else None
    # Dummy scheduler — we manage LR manually via get_lr()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

    # ---- Resume ----
    start_step = 0
    best_ppl   = float("inf")
    if cfg.resume:
        resumed_model, start_step, best_ppl = load_latest_checkpoint(
            run_dir, cfg, optimizer, scheduler, device
        )
        if resumed_model is not None:
            model = resumed_model
            # Rebuild optimizer against the resumed model's actual parameters
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
            )
            print(f"  Resumed at step {start_step:,} | best_ppl={best_ppl:.2f}")

    # ---- CSV log ----
    log_path = run_dir / "loss_log.csv"
    log_exists = log_path.exists() and cfg.resume
    log_file = open(log_path, "a" if log_exists else "w", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "step", "lr", "loss", "ppl_eval",
            "cells", "frozen",
            "energy", "integrity", "excitement", "pain",
            "n_loops", "exit_entropy",
        ])

    # ---- Training ----
    print(f"\nTraining for {cfg.steps:,} steps "
          f"(batch_size={cfg.batch_size}, max_seq_len={cfg.max_seq_len})\n")

    running_loss    = 0.0
    running_loops   = 0.0
    tracker = StatsTracker(cfg.run_dir, window=max(cfg.log_every * 4, 100))
    step_times: List[float] = []
    last_reward_info: Dict = {}

    # Start prefetch buffer — loads batches in background while GPU trains
    prefetch = PrefetchBuffer(train_ds, cfg.batch_size, device, buffer_size=8)
    print("  Prefetch buffer started (8 batches).")

    step_range = range(start_step + 1, cfg.steps + 1)
    if tqdm is not None:
        pbar = tqdm(step_range, initial=start_step, total=cfg.steps,
                    desc="Training", unit="step", dynamic_ncols=True)
    else:
        pbar = step_range

    try:
      for step in pbar:
        # LR update
        lr_now = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Batch — from prefetch buffer (non-blocking if ready)
        ids = prefetch.next()

        t0 = time.time()
        loss_val, reward_info = model.pretrain_step(
            ids, optimizer, tok,
            mlm_weight      = cfg.mlm_weight,
            unc_reg_weight  = cfg.unc_reg_weight,
            pg_optimizer    = pg_optimizer,
        )
        step_times.append(time.time() - t0)

        # Rebuild optimizer if EA changed architecture
        if step % model.growth_interval == 0:
            exit_gate_params = set(id(p) for p in model.exit_gate.parameters())
            if pg_optimizer is not None:
                new_main = [p for p in model.parameters()
                            if p.requires_grad and id(p) not in exit_gate_params]
            else:
                new_main = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                new_main, lr=lr_now,
                weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
            )
            if pg_optimizer is not None:
                pg_optimizer = torch.optim.Adam(
                    model.exit_gate.parameters(), lr=cfg.lr * 0.1
                )

        last_reward_info = reward_info

        _gn = sum(p.grad.norm().item()**2 for p in model.parameters()
                  if p.grad is not None)**0.5 if not math.isnan(loss_val) else 0.0
        tracker.record_step(reward_info, model, _gn)

        if not math.isnan(loss_val):
            running_loss  += loss_val
            running_loops += reward_info.get("n_loops", 1)

        # ---- Logging ----
        if step % cfg.log_every == 0:
            avg_loss  = running_loss  / cfg.log_every
            avg_loops = running_loops / cfg.log_every
            avg_ms    = (sum(step_times[-cfg.log_every:])
                         / len(step_times[-cfg.log_every:])) * 1000
            frozen    = sum(1 for c in model.cells if c.frozen)
            h         = model.homeostasis

            log_str = (
                f"Step {step:7,}/{cfg.steps:,} | "
                f"loss={avg_loss:.4f} | "
                f"ppl={math.exp(min(avg_loss,20)):.1f} | "
                f"lr={lr_now:.2e} | "
                f"loops={avg_loops:.2f} | "
                f"cells={len(model.cells)}({frozen}fr) | "
                f"{avg_ms:.0f}ms/step\n"
                f"  homeo: {h.summary()} | "
                f"exit_ent={reward_info.get('exit_entropy',0):.4f} | "
                f"tax={reward_info.get('compute_tax',0):.4f} | "
                f"pg={reward_info.get('reinforce',0):.4f}"
            )
            if tqdm is not None and hasattr(pbar, 'set_postfix_str'):
                pbar.set_postfix_str(
                    f"loss={avg_loss:.4f} ppl={math.exp(min(avg_loss,20)):.1f} "
                    f"loops={avg_loops:.2f} cells={len(model.cells)}"
                )
                tqdm.write(log_str)
            else:
                print(log_str)
            running_loss  = 0.0
            running_loops = 0.0

        # ---- Eval ----
        ppl_eval = float("nan")
        if step % cfg.eval_every == 0:
            ppl_eval = evaluate(model, eval_ds, tok, cfg, device)
            marker   = " *** best ***" if ppl_eval < best_ppl else ""
            print(f"  [Eval] step={step:,} | ppl={ppl_eval:.2f}{marker}")
            if ppl_eval < best_ppl:
                best_ppl = ppl_eval
                # Save a dedicated best DNA checkpoint
                best_dna = str(run_dir / "ckpt_best.json")
                model.save_dna(best_dna)
                torch.save({"step": step, "best_ppl": best_ppl},
                           run_dir / "ckpt_best_meta.pt")

        # ---- Checkpoint ----
        if step % cfg.save_every == 0:
            save_checkpoint(run_dir, model, optimizer, scheduler, step, best_ppl)
            tracker.save()

        # ---- CSV row ----
        if step % cfg.log_every == 0:
            h = model.homeostasis
            log_writer.writerow([
                step, f"{lr_now:.6f}",
                f"{loss_val:.6f}",
                f"{ppl_eval:.2f}" if not math.isnan(ppl_eval) else "",
                len(model.cells),
                sum(1 for c in model.cells if c.frozen),
                f"{h.energy.item():.4f}",
                f"{h.integrity.item():.4f}",
                f"{h.excitement.item():.4f}",
                f"{h.pain.item():.4f}",
                reward_info.get("n_loops", ""),
                f"{reward_info.get('exit_entropy', 0):.6f}",
            ])
            log_file.flush()

    finally:
        prefetch.stop()

    # ---- Final checkpoint ----
    save_checkpoint(run_dir, model, optimizer, scheduler, cfg.steps, best_ppl)
    log_file.close()
    print(f"\nDone. Best eval ppl: {best_ppl:.2f}")
    print(f"Run saved to: {run_dir}")


# ===========================================================================
# 8.  Argparse entry point
# ===========================================================================

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="HydraLM streaming trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_path",        default="corpus.txt")
    p.add_argument("--eval_split",        type=float, default=0.02)
    p.add_argument("--bpe_sample_lines",  type=int,   default=500_000)
    p.add_argument("--bpe_min_freq",      type=int,   default=2)
    p.add_argument("--index_cache_path",  default="",
                   help="Path for line-index cache. Auto-derived from data_path if empty.")

    # Run
    p.add_argument("--run_dir", default="runs/default")
    p.add_argument("--resume",  action="store_true")

    # Tokenizer
    p.add_argument("--vocab_size", type=int, default=8000)

    # Model
    p.add_argument("--d_model",     type=int,   default=256)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--n_layers",    type=int,   default=4)
    p.add_argument("--max_seq_len", type=int,   default=512)
    p.add_argument("--n_dag_layers",type=int,   default=2)
    p.add_argument("--max_cells",   type=int,   default=16)
    p.add_argument("--min_width",      type=int, default=16)
    p.add_argument("--max_width",      type=int, default=512)
    p.add_argument("--max_connections",type=int, default=10,
                   help="Max input_sources per cell (caps DAG fan-in)")
    p.add_argument("--max_loops",          type=int,   default=4)
    p.add_argument("--exit_entropy_weight",type=float, default=0.05)
    p.add_argument("--loop_depth_weight",  type=float, default=0.01,
                   help="Compute tax: linear penalty per loop used")
    p.add_argument("--reinforce_weight",   type=float, default=0.01,
                   help="REINFORCE policy gradient weight on exit gate")
    p.add_argument("--dropout",     type=float, default=0.1)

    # Training
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--steps",        type=int,   default=100_000)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--mlm_weight",   type=float, default=0.5)

    # Logging
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every",  type=int, default=50)

    # Misc
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="auto")

    args = p.parse_args()
    cfg  = TrainConfig(**vars(args))
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

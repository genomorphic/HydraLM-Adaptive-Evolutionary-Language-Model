"""
HydraLM — Adaptive Language Model
===================================
Stripped from HydraAdaptive v3 / Phase 1-2 for pure language modelling.

Key components
--------------
  SuperBPETokenizer    : byte-level BPE tokenizer (no external dependencies).
                         Learns merge rules from a corpus; handles arbitrary UTF-8.
  TextEncoder          : Transformer encoder with per-token UncertaintyGate.
  HydraAdaptiveCell    : Evolutionary cell with Shapley health attribution.
  RelativeEA           : Tournament-style population pruning + regrowth.
  LanguageHead         : Autoregressive next-token projection.
  Homeostasis          : Four-scalar internal state driving intrinsic rewards.
  HydraLM              : Full assembled model with pretraining + generation.

Homeostasis → Intrinsic Rewards (pretraining)
----------------------------------------------
  energy      → curiosity bonus:     upweight high-perplexity tokens when energy
                                      is low (model is "tired" of easy patterns).
  excitement  → exploration bonus:   entropy regularisation on token distribution
                                      scaled by novelty/surprise signal.
  pain        → consolidation clip:  reduce gradient contribution of the hardest
                                      tokens when pain is high (stabilise on outliers).
  integrity   → depth gate:          scale UncertaintyGate routing thresholds —
                                      low integrity forces deeper processing paths.

SuperBPE specifics
------------------
  - Byte-level base vocabulary (256 tokens) — any UTF-8 handled natively.
  - BPE merge learning runs on raw byte sequences.
  - Special tokens: PAD=256, MASK=257, BOS=258, EOS=259 → vocab_size=260+merges.
  - Serializable: save/load merge rules as JSON.
  - Vocabulary can be extended post-hoc (new merges appended).

Quick start
-----------
  from hydra_lm import HydraLM, SuperBPETokenizer

  # Build and train tokenizer
  tok = SuperBPETokenizer(vocab_size=1000)
  tok.train(["Hello world", "This is a test", ...])

  # Build model
  model = HydraLM(vocab_size=tok.vocab_size)

  # Pretrain (MLM + autoregressive)
  loss, rewards = model.pretrain_step(token_ids, optimizer)

  # Generate
  ids = model.generate(prompt_ids, max_new=64)
  text = tok.decode(ids[0].tolist())
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1.  SuperBPE Tokenizer
# ===========================================================================

class SuperBPETokenizer:
    """
    Byte-level BPE tokenizer — no external dependencies.

    Base vocabulary
    ---------------
    Indices 0-255 : individual bytes (covers all UTF-8 natively).
    Index   256   : <PAD>
    Index   257   : <MASK>
    Index   258   : <BOS>
    Index   259   : <EOS>
    Indices 260+  : learned BPE merge tokens.

    Training
    --------
    Follows Sennrich et al. (2016) BPE algorithm on byte sequences:
      1. Encode each training text as a sequence of bytes.
      2. Count all adjacent byte-pair frequencies.
      3. Merge the most frequent pair into a new token.
      4. Repeat until vocab_size is reached.

    The merge table is a list of (token_a, token_b) -> merged_id tuples.
    Encoding applies merges greedily in the order they were learned.

    Serialisation
    -------------
    tok.save("tokenizer.json")   # saves vocab + merge rules
    tok = SuperBPETokenizer.load("tokenizer.json")
    """

    PAD_ID  = 256
    MASK_ID = 257
    BOS_ID  = 258
    EOS_ID  = 259
    _N_SPECIAL = 4      # PAD, MASK, BOS, EOS
    _BASE_VOCAB = 256   # byte values 0-255

    def __init__(self, vocab_size: int = 8000):
        """
        Args
            vocab_size : total vocabulary size including base bytes + specials
                         + learned merges. Must be >= 260.
        """
        if vocab_size < self._BASE_VOCAB + self._N_SPECIAL:
            raise ValueError(f"vocab_size must be >= {self._BASE_VOCAB + self._N_SPECIAL}")
        self.target_vocab_size = vocab_size

        # Merge table: list of (id_a, id_b) in learning order.
        # The merged token receives id = 260 + merge_index.
        self._merges: List[Tuple[int, int]] = []

        # Fast-lookup dict: (id_a, id_b) -> merged_id
        self._merge_map: Dict[Tuple[int, int], int] = {}

        # Reverse: merged_id -> (id_a, id_b) for decoding
        self._split_map: Dict[int, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Vocabulary size (dynamic — grows as merges are added)
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._BASE_VOCAB + self._N_SPECIAL + len(self._merges)

    @property
    def n_merges_needed(self) -> int:
        return self.target_vocab_size - self._BASE_VOCAB - self._N_SPECIAL

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts:         List[str],
        min_frequency: int  = 2,
        verbose:       bool = True,
        max_word_types: int = 100_000,  # cap unique word types fed to BPE
    ) -> None:
        """
        Learn BPE merge rules from a list of text strings.

        Performance
        -----------
        The BPE merge loop is O(unique_word_types) per merge, not O(corpus).
        Real English text has millions of unique word types but the vast
        majority are rare. We keep only the top `max_word_types` by frequency
        — these drive 99%+ of all merges. Rare words contribute negligibly to
        merge decisions but dominate vocabulary size and slow everything down.

        100k word types → ~2-5 minutes on CPU for 7740 merges.
        Quality vs 500k types: negligible difference in practice.
        """
        import time as _time

        n_merges = self.n_merges_needed
        if n_merges <= 0:
            return

        # ---- Build word frequency table ----
        if verbose:
            print(f"  [BPE] Building vocabulary from {len(texts):,} texts...")

        word_freqs: Counter = Counter()
        for text in texts:
            words = text.split()
            for i, w in enumerate(words):
                prefix = " " if i > 0 else ""
                bseq = tuple(b for b in (prefix + w).encode("utf-8"))
                word_freqs[bseq] += 1

        # Keep only the top max_word_types most frequent word types.
        # Rare hapax legomena don't influence merge decisions but blow up
        # the vocabulary size and make every iteration slow.
        if len(word_freqs) > max_word_types:
            top_words = word_freqs.most_common(max_word_types)
            word_freqs = Counter(dict(top_words))
            if verbose:
                print(f"  [BPE] Keeping top {max_word_types:,} word types "
                      f"(of {len(word_freqs):,} unique)")

        # Working vocab: word_tuple -> token list (mutable)
        vocab: Dict[Tuple[int, ...], List[int]] = {
            seq: list(seq) for seq in word_freqs
        }

        # ---- Build initial pair counts + reverse index ----
        pair_freqs:    Dict[Tuple[int,int], int] = {}
        pair_to_words: Dict[Tuple[int,int], set] = {}

        for word_seq, tokens in vocab.items():
            freq = word_freqs[word_seq]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word_seq)

        log_interval = max(1, n_merges // 10)
        t0 = _time.time()

        for merge_idx in range(n_merges):
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            best_freq = pair_freqs[best_pair]
            if best_freq < min_frequency:
                break

            new_id = self._BASE_VOCAB + self._N_SPECIAL + merge_idx
            self._merges.append(best_pair)
            self._merge_map[best_pair] = new_id
            self._split_map[new_id]    = best_pair

            a, b = best_pair
            affected = pair_to_words.pop(best_pair, set())

            for word_seq in affected:
                if word_seq not in vocab:
                    continue
                tokens  = vocab[word_seq]
                freq    = word_freqs[word_seq]
                new_tok: List[int] = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                        if i > 0:
                            old_left = (tokens[i-1], tokens[i])
                            pair_freqs[old_left] = pair_freqs.get(old_left, 0) - freq
                            if pair_freqs[old_left] <= 0:
                                pair_freqs.pop(old_left, None)
                                pair_to_words.pop(old_left, None)
                            elif old_left in pair_to_words:
                                pair_to_words[old_left].discard(word_seq)
                        if i < len(tokens)-2:
                            old_right = (tokens[i+1], tokens[i+2])
                            pair_freqs[old_right] = pair_freqs.get(old_right, 0) - freq
                            if pair_freqs[old_right] <= 0:
                                pair_freqs.pop(old_right, None)
                                pair_to_words.pop(old_right, None)
                            elif old_right in pair_to_words:
                                pair_to_words[old_right].discard(word_seq)
                        new_tok.append(new_id)
                        i += 2
                    else:
                        new_tok.append(tokens[i])
                        i += 1

                vocab[word_seq] = new_tok

                for j in range(len(new_tok)-1):
                    if new_tok[j] == new_id or new_tok[j+1] == new_id:
                        new_pair = (new_tok[j], new_tok[j+1])
                        pair_freqs[new_pair] = pair_freqs.get(new_pair, 0) + freq
                        if new_pair not in pair_to_words:
                            pair_to_words[new_pair] = set()
                        pair_to_words[new_pair].add(word_seq)

            pair_freqs.pop(best_pair, None)

            if verbose and (merge_idx + 1) % log_interval == 0:
                pct  = (merge_idx + 1) / n_merges * 100
                secs = _time.time() - t0
                eta  = secs / (merge_idx + 1) * (n_merges - merge_idx - 1)
                print(f"  [BPE] {merge_idx+1}/{n_merges} merges ({pct:.0f}%) "
                      f"| vocab={self.vocab_size} "
                      f"| freq={best_freq} "
                      f"| {secs/60:.1f}min | ETA {eta/60:.1f}min",
                      flush=True)

        if verbose:
            elapsed = _time.time() - t0
            print(f"  [BPE] Training complete in {elapsed/60:.1f}min. "
                  f"Vocab size: {self.vocab_size} "
                  f"({len(self._merges)} merges learned)")

    @staticmethod
    def _apply_merge(
        vocab: Dict[Tuple[int, ...], int],
        pair:  Tuple[int, int],
        new_id: int,
    ) -> Dict[Tuple[int, ...], int]:
        """Replace all occurrences of pair in vocab sequences with new_id."""
        a, b = pair
        new_vocab: Dict[Tuple[int, ...], int] = {}
        for seq, freq in vocab.items():
            new_seq: List[int] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_vocab[tuple(new_seq)] = freq
        return new_vocab

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode a string to a list of token IDs.

        Applies learned merges greedily in training order.
        Unknown byte values are impossible since all 256 bytes are in the
        base vocabulary.
        """
        # Byte-level base encoding
        ids: List[int] = list(text.encode("utf-8"))

        # Apply merges greedily (standard BPE encoding)
        for pair, new_id in self._merge_map.items():
            ids = self._apply_merge_to_ids(ids, pair[0], pair[1], new_id)

        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    @staticmethod
    def _apply_merge_to_ids(
        ids: List[int], a: int, b: int, new_id: int
    ) -> List[int]:
        result: List[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = 256,
        padding: bool = True,
        add_bos: bool = False,
        add_eos: bool = False,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Encode a batch of strings to a (B, max_length) int64 tensor.
        Sequences shorter than max_length are padded with PAD_ID.
        """
        encoded = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
            for t in texts
        ]
        if padding:
            max_len = max(len(e) for e in encoded) if encoded else max_length
            max_len = min(max_len, max_length)
            padded = [
                e[:max_len] + [self.PAD_ID] * max(0, max_len - len(e))
                for e in encoded
            ]
        else:
            padded = encoded
        return torch.tensor(padded, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        Skips special tokens (PAD, MASK, BOS, EOS).
        """
        # Recursively expand merged tokens to their constituent bytes
        bytes_out: List[int] = []
        for tok_id in ids:
            if tok_id in (self.PAD_ID, self.MASK_ID, self.BOS_ID, self.EOS_ID):
                continue
            bytes_out.extend(self._expand_to_bytes(tok_id))
        # Decode byte sequence as UTF-8 with error replacement
        return bytes(bytes_out).decode("utf-8", errors="replace")

    def _expand_to_bytes(self, tok_id: int) -> List[int]:
        """Recursively expand a token to its constituent bytes."""
        if tok_id < 256:
            return [tok_id]
        if tok_id in self._split_map:
            a, b = self._split_map[tok_id]
            return self._expand_to_bytes(a) + self._expand_to_bytes(b)
        return []  # unknown token

    # ------------------------------------------------------------------
    # Masking (for MLM pretraining)
    # ------------------------------------------------------------------

    def apply_mask(
        self,
        ids: torch.Tensor,      # (B, L) int64
        mask_prob: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask mask_prob fraction of non-special tokens.
        Returns (masked_ids, target_ids) where target_ids has -100
        for un-masked positions (ignored by cross_entropy).
        """
        masked  = ids.clone()
        targets = ids.clone().fill_(-100)

        special = {self.PAD_ID, self.MASK_ID, self.BOS_ID, self.EOS_ID}
        is_special = torch.zeros_like(ids, dtype=torch.bool)
        for sid in special:
            is_special |= (ids == sid)

        prob_matrix = torch.rand_like(ids.float())
        mask_bool   = (prob_matrix < mask_prob) & ~is_special

        masked[mask_bool]  = self.MASK_ID
        targets[mask_bool] = ids[mask_bool]
        return masked, targets

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "target_vocab_size": self.target_vocab_size,
            "merges": self._merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  [BPE] Saved tokenizer to {path} ({len(self._merges)} merges)")

    @classmethod
    def load(cls, path: str) -> "SuperBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["target_vocab_size"])
        for merge_idx, (a, b) in enumerate(data["merges"]):
            pair   = (a, b)
            new_id = cls._BASE_VOCAB + cls._N_SPECIAL + merge_idx
            tok._merges.append(pair)
            tok._merge_map[pair]   = new_id
            tok._split_map[new_id] = pair
        print(f"  [BPE] Loaded tokenizer from {path} "
              f"(vocab={tok.vocab_size}, {len(tok._merges)} merges)")
        return tok


# ===========================================================================
# 2.  UncertaintyGate (from uncertainty_gate.py, inlined for single-file dist)
# ===========================================================================

class _UncertaintyProbe(nn.Module):
    def __init__(self, in_dim: int, probe_dim: int = 16, n_probes: int = 5):
        super().__init__()
        self.probes = nn.ModuleList([
            nn.Linear(in_dim, probe_dim, bias=False) for _ in range(n_probes)
        ])
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs  = torch.stack([p(x) for p in self.probes], dim=1)  # B, K, D
        # correction=0 (population var) avoids NaN when K==1 or B==1
        variance = outputs.var(dim=1, correction=0).mean(dim=-1)     # B
        variance = variance.clamp(0.0, 10.0)                         # guard explosion
        return torch.sigmoid(variance * F.softplus(self.temperature))


class _DepthStage(nn.Module):
    def __init__(self, channels: int, expansion: int = 2,
                 stage_idx: int = 0, n_stages: int = 4):
        super().__init__()
        mid = channels * expansion
        self.expand    = nn.Linear(channels, mid)
        self.depthwise = nn.Conv1d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.contract  = nn.Linear(mid, channels)
        self.norm      = nn.LayerNorm(channels)
        self.ara_a     = nn.Parameter(torch.ones(1))
        self.ara_beta  = nn.Parameter(torch.ones(1) * 1.5)
        init_thresh    = stage_idx / max(n_stages - 1, 1)
        self.threshold = nn.Parameter(torch.tensor(init_thresh))

    def ara(self, x):
        a = F.softplus(self.ara_a)
        b = F.softplus(self.ara_beta)
        return x + a * torch.tanh(b * x)

    def gate(self, unc):
        # unc can be (B,) sequence-level or (B, S) token-level
        return torch.sigmoid((unc - torch.sigmoid(self.threshold)) / 0.1)

    def forward(self, x, unc=None):
        # x: (B, S, C)
        # unc: (B, S) per-token uncertainty — each token gates independently
        B, S, C = x.shape
        if unc is None:
            unc = torch.full((B, S), 0.5, device=x.device)
        elif unc.dim() == 1:
            # Legacy (B,) sequence-level — broadcast to (B, S)
            unc = unc.unsqueeze(1).expand(B, S)
        g = self.gate(unc).unsqueeze(-1)   # (B, S, 1) — per-token gate
        h = self.ara(self.expand(x))
        h = self.depthwise(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.contract(h)
        eff_gate = 0.05 + 0.95 * g        # (B, S, 1) broadcasts over C
        return self.norm(x + eff_gate * h)


class UncertaintyGateSeq(nn.Module):
    """
    Uncertainty-modulated adaptive depth block for (B, S, C) sequences.
    Each token's compute depth is determined by its epistemic uncertainty.
    integrity_scale allows Homeostasis to force deeper paths when integrity
    is low (the model is "damaged" and needs more careful processing).
    """

    def __init__(self, channels: int, n_stages: int = 4,
                 n_probes: int = 5, probe_dim: int = 16,
                 expansion: int = 2, gate_warmup_steps: int = 500):
        super().__init__()
        self.gate_warmup_steps = gate_warmup_steps
        self.probe  = _UncertaintyProbe(channels, probe_dim, n_probes)
        self.stages = nn.ModuleList([
            _DepthStage(channels, expansion, i, n_stages)
            for i in range(n_stages)
        ])
        self.bypass_proj = nn.Linear(channels, channels)
        self.out_norm    = nn.LayerNorm(channels)
        self.blend_bias  = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x:               torch.Tensor,   # (B, S, C)
        global_step:     int   = 999999,
        integrity_scale: float = 1.0,    # from Homeostasis.integrity
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (output, uncertainty).
        integrity_scale in [0,1]: lower → push deeper stage thresholds down
        so more tokens get routed through deeper computation.
        """
        if global_step < self.gate_warmup_steps:
            return x, torch.zeros(x.shape[:2], device=x.device)

        B, S, C = x.shape
        flat    = x.reshape(B * S, C)
        unc_flat = self.probe(flat)                       # (B*S,)
        unc      = unc_flat.reshape(B, S)

        # Integrity scaling: low integrity lowers effective thresholds
        # so tokens are routed through deeper stages more aggressively
        depth_scale = 2.0 - integrity_scale              # 1.0 at full integrity, 2.0 at zero
        scaled_unc  = (unc * depth_scale).clamp(0.0, 1.0)

        bypassed = x + self.bypass_proj(x) * 0.1

        # Pass per-token uncertainty (B, S) directly to each stage.
        # Each token routes through deep stages independently based on its
        # own epistemic uncertainty — the fix GPT correctly identified.
        deep = x
        for stage in self.stages:
            deep = stage(deep, scaled_unc)   # (B, S) not (B,)

        # Blend: per-token blend coefficient from per-token uncertainty
        # (B, S) -> (B, S, 1) for broadcasting over channels
        blend = torch.sigmoid(scaled_unc + self.blend_bias).unsqueeze(-1)
        safe  = blend.clamp(0.1, 0.9)        # (B, S, 1)
        out   = self.out_norm(safe * deep + (1.0 - safe) * bypassed)
        return out, unc

    def uncertainty_reg_loss(self, unc: torch.Tensor) -> torch.Tensor:
        # Maximise entropy over uncertainty: push u toward 0.5 (neither certain nor uncertain).
        # Loss = negative entropy → minimising this maximises entropy.
        u       = unc.clamp(0.01, 0.99)   # tighter clamp keeps log well-behaved
        entropy = -(u * u.log() + (1.0 - u) * (1.0 - u).log())
        return -entropy.mean()


# ===========================================================================
# 3.  Homeostasis
# ===========================================================================

class Homeostasis(nn.Module):
    """
    Four-scalar internal state used to generate intrinsic training rewards.

    Variables
    ---------
    energy     [0,1] : running capacity (drains each step, recovers on low loss).
    integrity  [0,1] : structural health (damaged by high loss, recovers slowly).
    excitement [0,1] : novelty signal (spikes on large loss changes).
    pain       [0,1] : acute distress (0=healthy, 1=maximum distress).

    Intrinsic rewards (computed in HydraLM.pretrain_step)
    -----------------------------------------------------
    curiosity_bonus   (energy)     : upweight loss on high-perplexity tokens
                                     when energy is low — push toward hard examples.
    exploration_bonus (excitement) : add entropy regularisation to encourage
                                     diverse predictions when excited.
    consolidation_clip (pain)      : reduce gradient from the hardest tokens
                                     when pain is high — avoid destabilisation.
    depth_pressure (integrity)     : scale UncertaintyGate thresholds lower when
                                     integrity is low — force deeper processing.
    """

    def __init__(
        self,
        energy_decay:       float = 0.002,
        energy_recovery:    float = 0.005,
        pain_decay:         float = 0.15,
        excitement_decay:   float = 0.10,
        pain_threshold:     float = 0.20,
        excitement_trigger: float = 0.05,
        warmup_steps:       int   = 200,    # steps before pain/integrity fire
    ):
        super().__init__()
        self.energy_decay       = energy_decay
        self.energy_recovery    = energy_recovery
        self.pain_decay         = pain_decay
        self.excitement_decay   = excitement_decay
        self.pain_threshold     = pain_threshold
        self.excitement_trigger = excitement_trigger
        self.warmup_steps       = warmup_steps

        self.register_buffer("energy",     torch.tensor(1.0))
        self.register_buffer("integrity",  torch.tensor(1.0))
        self.register_buffer("excitement", torch.tensor(0.0))
        # Pain lives in [0, 1]: 0 = no distress, 1 = maximum distress.
        # Positive to match the convention of the other three homeostasis scalars.
        self.register_buffer("pain",       torch.tensor(0.0))
        # clm_loss_ema tracks the CLM (next-token) loss only — the clean
        # language learning signal decoupled from regularisation terms.
        # All four homeostasis variables are computed against this, not
        # the composite training loss which includes MLM, entropy, tax etc.
        self.register_buffer("clm_loss_ema",  torch.tensor(100.0))
        self.register_buffer("homeo_step",    torch.tensor(0))

    def get_vector(self, batch_size: int, device) -> torch.Tensor:
        """
        Returns (B, 4) detached state vector: [energy, integrity, excitement, pain].
        All four scalars are in [0, 1].
        """
        energy     = self.energy.clamp(0.0, 1.0).to(device)
        integrity  = self.integrity.clamp(0.0, 1.0).to(device)
        excitement = self.excitement.clamp(0.0, 1.0).to(device)
        pain       = self.pain.clamp(0.0, 1.0).to(device)
        vec = torch.stack([energy, integrity, excitement, pain])
        return vec.detach().unsqueeze(0).expand(batch_size, -1)

    def update(self, clm_loss: float, prev_clm_loss: float) -> None:
        """
        Update homeostasis state from the CLM (next-token) loss only.

        Decoupled from total training loss deliberately — the composite
        loss includes MLM, regularisation, compute tax etc. which are
        optimisation signals, not language learning signals. Homeostasis
        should respond to whether the model is getting better at predicting
        text, not whether auxiliary objectives are being satisfied.

        clm_loss      : current step CLM cross-entropy loss (language signal)
        prev_clm_loss : previous step CLM loss (for regression detection)
        """
        with torch.no_grad():
            self.homeo_step += 1
            step = self.homeo_step.item()
            in_warmup = step <= self.warmup_steps

            # Update CLM EMA — converges faster during warmup
            alpha = 0.20 if in_warmup else 0.05
            self.clm_loss_ema = (1 - alpha) * self.clm_loss_ema + alpha * clm_loss
            baseline = self.clm_loss_ema.item() + 1e-8
            ratio    = clm_loss / baseline

            # Pain — suppressed during warmup
            if in_warmup:
                self.pain = torch.tensor(max(0.0, self.pain.item() * (1.0 - self.pain_decay)))
            else:
                regression      = clm_loss - prev_clm_loss
                regression_pain = max(0.0, regression / (abs(prev_clm_loss) + 1e-8))

                if ratio > 1.0 + self.pain_threshold:
                    # Severe: CLM loss well above its own baseline
                    acute = min(1.0, ratio - 1.0 - self.pain_threshold)
                    self.pain = torch.tensor(min(1.0, self.pain.item() + acute * 0.5))
                elif regression > 0.05 and regression_pain > 0.005 and ratio > 1.0:
                    # Mild: CLM loss increased vs previous step AND above baseline
                    mild = min(0.3, regression_pain * 2.0)
                    self.pain = torch.tensor(min(1.0, self.pain.item() + mild * 0.3))
                else:
                    self.pain = torch.tensor(max(0.0, self.pain.item() * (1.0 - self.pain_decay)))

            # Excitement — spikes on large CLM loss changes (novelty in language signal)
            delta = abs(clm_loss - prev_clm_loss)
            if delta > self.excitement_trigger:
                spike = min(1.0, delta / (self.excitement_trigger + 1e-8) * 0.2)
                self.excitement = torch.tensor(min(1.0, self.excitement.item() + spike))
            else:
                self.excitement = torch.tensor(
                    max(0.0, self.excitement.item() * (1.0 - self.excitement_decay))
                )

            # Energy — drains when CLM loss is above its own baseline
            drain    = self.energy_decay * (1.0 + self.pain.item())
            recovery = self.energy_recovery if clm_loss < baseline else 0.0
            self.energy = torch.tensor(
                max(0.0, min(1.0, self.energy.item() - drain + recovery))
            )

            # Integrity — tracks CLM improvement trend, suppressed during warmup
            if not in_warmup:
                if ratio < 0.98:
                    self.integrity = torch.tensor(min(1.0, self.integrity.item() + 0.005))
                elif ratio > 1.30:
                    self.integrity = torch.tensor(max(0.0, self.integrity.item() - 0.003))
                elif (clm_loss - prev_clm_loss) > 0.05:
                    self.integrity = torch.tensor(max(0.0, self.integrity.item() - 0.001))

    def intrinsic_rewards(self) -> Dict[str, float]:
        """
        Returns a dict of named intrinsic reward scalars for the current step.
        These are applied inside HydraLM.pretrain_step to shape the loss.
        """
        e   = self.energy.item()
        ex  = self.excitement.item()
        p   = self.pain.item()      # in [0, 1]
        ing = self.integrity.item()

        return {
            # curiosity: low energy → push harder, upweight high-perplexity tokens
            # range [0.0, 1.0] — higher when tired
            "curiosity":     1.0 - e,

            # exploration: excitement → diversify predictions
            # range [0.0, 0.10]
            "exploration":   ex * 0.10,

            # consolidation: pain drives the clip fraction directly [0, 0.5].
            "consolidation": p * 0.5,

            # depth_pressure: low integrity → force deeper processing
            # passed as integrity_scale to UncertaintyGateSeq
            # range [0.0, 1.0]
            "depth_pressure": ing,

            # raw pain for logging
            "pain_raw": p,
        }

    def summary(self) -> str:
        return (
            f"energy={self.energy.item():.3f} "
            f"integrity={self.integrity.item():.3f} "
            f"excitement={self.excitement.item():.3f} "
            f"pain={self.pain.item():.3f}"   # 0=healthy, 1=max distress
        )


# ===========================================================================
# 4.  Evolutionary Cell Population
# ===========================================================================

class HydraAdaptiveCell(nn.Module):
    """Base evolutionary cell with ARA activation and Hebbian plasticity."""

    def __init__(self, in_features: int, out_features: int,
                 plasticity_lr: float = 5e-4, health_decay: float = 0.995,
                 layer_idx: int = 0, input_sources: list = None):
        super().__init__()
        self.in_features    = in_features
        self.out_features   = out_features
        self.plasticity_lr  = plasticity_lr
        self.health_decay   = health_decay
        self.layer_idx      = layer_idx
        # Explicit list of cell indices whose outputs feed into this cell.
        # Empty list = root cell (takes encoder+homeo only).
        self.input_sources: List[int] = list(input_sources) if input_sources is not None else []

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.beta = nn.Parameter(torch.ones(out_features))
        self.freq = nn.Parameter(torch.empty(out_features).uniform_(0.8, 1.2))

        self.register_buffer("health",             torch.tensor(1.0))
        self.register_buffer("age",                torch.tensor(0.0))
        self.register_buffer("high_health_streak", torch.tensor(0.0))
        self.frozen: bool = False

    def freeze(self):
        for p in (self.weight, self.bias, self.beta, self.freq):
            p.requires_grad_(False)
        self.frozen = True

    def unfreeze(self):
        for p in (self.weight, self.bias, self.beta, self.freq):
            p.requires_grad_(True)
        self.frozen = False

    def forward(self, x: torch.Tensor, plastic: bool = False) -> torch.Tensor:
        pre = F.linear(x, self.weight, self.bias)
        out = torch.tanh(self.beta * pre * self.freq)
        self.age += 1.0
        if plastic and not self.training:
            with torch.no_grad():
                dw = (out.t() @ x) / max(x.size(0), 1)
                self.weight.add_(dw, alpha=self.plasticity_lr)
                self.weight.mul_(1.0 - 1e-3)
        return out

    def tick_health_streak(self, new_health: float, threshold: float,
                           needed: int) -> bool:
        with torch.no_grad():
            self.health = torch.clamp(torch.tensor(new_health), 0.0, 1.0)
            if new_health >= threshold:
                self.high_health_streak += 1.0
            else:
                self.high_health_streak.fill_(0.0)
        return self.high_health_streak.item() >= needed


class ShapleyAttributor:
    """
    Path-aware permutation-sampling Shapley values over the full cell DAG.

    Why path-aware matters
    ----------------------
    The original implementation only ran Shapley on L0 (root) cells, treating
    them as independent. With a DAG topology, deeper cells take the outputs of
    upstream cells as inputs. If cell 3 feeds cell 5, removing cell 3 from a
    subset correctly degrades cell 5's output (it receives zeros instead), so
    cell 3 gets credit for its downstream contribution.

    Algorithm
    ---------
    For each permutation sample:
      1. Build a topological order over all n cells.
      2. Walk the order; for each prefix subset, run a DAG forward pass where:
         - Cells IN the subset execute normally.
         - Cells NOT in the subset contribute zero tensors to their consumers.
      3. Aggregate all active cell outputs (health-gated), project via
         language_head, compute cross-entropy against targets.
      4. Marginal contribution of adding cell i = loss_before - loss_after.

    Complexity
    ----------
    O(n_samples × n × n) forward passes — each sample requires n incremental
    evals, each eval is O(n) DAG traversal. With n_samples=6 and n=24 cells
    this is 144 mini forward passes per _evolve call. Kept cheap by using
    the detached enc tensor (no encoder recomputation) and @torch.no_grad().
    """

    def __init__(self, n_samples: int = 6):
        self.n_samples = n_samples

    @torch.no_grad()
    def compute(
        self,
        cells:        list,           # all HydraAdaptiveCell in population
        adapters:     list,           # paired nn.Linear adapters
        enc:          torch.Tensor,   # (B, enc_dim+homeo) fused encoder+homeo
        y:            torch.Tensor,   # (B,) next-token targets
        output_layer: nn.Module,      # language_head
        hidden_dim:   int,
        pad_id:       int = 256,
    ) -> List[float]:
        """
        Returns list of Shapley values, one per cell, same order as cells.
        Values are non-negative (floored at 0); normalise externally.
        """
        n = len(cells)
        if n == 1:
            return [1.0]

        # Build topological order once (shared across all samples)
        topo = self._topo_order(cells, n)
        shapley = [0.0] * n

        for _ in range(self.n_samples):
            # Random permutation of cell indices
            perm = list(range(n))
            random.shuffle(perm)

            # Evaluate with empty subset first
            prev_loss = self._eval_dag_subset(
                set(), topo, cells, adapters, enc, y,
                output_layer, hidden_dim, pad_id
            )

            active: set = set()
            for idx in perm:
                active.add(idx)
                curr_loss = self._eval_dag_subset(
                    active, topo, cells, adapters, enc, y,
                    output_layer, hidden_dim, pad_id
                )
                shapley[idx] += prev_loss - curr_loss
                prev_loss = curr_loss

        return [max(0.0, s / self.n_samples) for s in shapley]

    @staticmethod
    def _topo_order(cells, n: int) -> List[int]:
        """Kahn topological sort — same logic as HydraLM._topo_order."""
        in_deg = [0] * n
        adj    = [[] for _ in range(n)]
        for i, cell in enumerate(cells):
            for src in cell.input_sources:
                if 0 <= src < n:
                    adj[src].append(i)
                    in_deg[i] += 1
        queue = [i for i in range(n) if in_deg[i] == 0]
        order: List[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
        return order if len(order) == n else list(range(n))

    def _eval_dag_subset(
        self,
        active:       set,            # indices of cells included in this eval
        topo:         List[int],      # topological order
        cells:        list,
        adapters:     list,
        enc:          torch.Tensor,   # (B, base_in) — root cell input
        y:            torch.Tensor,   # (B,) targets
        output_layer: nn.Module,
        hidden_dim:   int,
        pad_id:       int,
    ) -> float:
        """
        Run DAG forward pass with only `active` cells executing.
        Inactive cells contribute zero tensors to their consumers.
        Returns cross-entropy loss.
        """
        B      = enc.size(0)
        device = enc.device

        # cell_outs[i] = adapter(cell(input)) if i in active, else zeros
        cell_outs: Dict[int, torch.Tensor] = {}

        for idx in topo:
            if idx not in active:
                cell_outs[idx] = torch.zeros(B, hidden_dim, device=device)
                continue

            # Build input: enc + source outputs (zeros if source inactive)
            sources = cells[idx].input_sources
            if not sources:
                x_in = enc
            else:
                src_tensors = [
                    cell_outs.get(s, torch.zeros(B, hidden_dim, device=device))
                    for s in sources
                ]
                # enc here may be wider than cell expects if loop proj was used;
                # use only the base_in slice to match cell's in_features
                base_in = cells[idx].in_features - len(sources) * hidden_dim
                x_in = torch.cat([enc[:, :base_in]] + src_tensors, dim=1)

            try:
                out = adapters[idx](cells[idx](x_in))
            except RuntimeError:
                # Shape mismatch on a stale cell — treat as zero contribution
                out = torch.zeros(B, hidden_dim, device=device)
            cell_outs[idx] = out

        # Health-gated aggregate over active cells only
        active_outs = [cell_outs[i] for i in range(len(cells)) if i in active]
        if not active_outs:
            agg = torch.zeros(B, hidden_dim, device=device)
        else:
            health_scores = torch.stack([
                cells[i].health.to(device)
                for i in range(len(cells)) if i in active
            ])
            gates   = F.softmax(health_scores.float(), dim=0)
            stacked = torch.stack(active_outs, dim=0)               # (k, B, H)
            agg     = (stacked * gates.view(-1, 1, 1)).sum(0)       # (B, H)

        logits = output_layer(agg)
        mask   = (y != pad_id)
        if mask.sum() == 0:
            return 0.0
        return F.cross_entropy(logits[mask], y[mask]).item()


# ===========================================================================
# 4a.  Gene Pool  (persistent weight inheritance bank)
# ===========================================================================

class GenePool:
    """
    Persistent ring-buffer of weight snapshots from high-contribution cells.

    Cells donate on two events:
      • Death during pruning  — if the cell was in the elite set at the time
                                it was selected for pruning (useful but evicted).
      • Freeze               — cell has proven long-term utility.

    Flash modes (randomly assigned 1/3 each to non-crossover spawns):
      • DIRECT    — truncate/pad from the closest-shape donor in the pool.
      • AVERAGE   — average the overlap submatrix from all same-layer donors.
      • SUBSPACE  — SVD the best donor, project top-k singular vectors into
                    the child's weight space.

    Crossover spawns (elite×elite) are never touched by the gene pool.

    Storage
    -------
    Each snapshot: {"w": Tensor(out, in), "layer": int, "out": int, "in": int}
    Stored on CPU regardless of training device.
    Cap: 200 snapshots (ring — oldest evicted when full).
    """

    CAPACITY  = 200
    MODE_DIRECT   = 0
    MODE_AVERAGE  = 1
    MODE_SUBSPACE = 2

    def __init__(self):
        self._pool: List[Dict] = []   # ring buffer
        self._ptr:  int        = 0    # next write position

    # ------------------------------------------------------------------
    # Donation
    # ------------------------------------------------------------------

    def donate(self, cell: "HydraAdaptiveCell", source: str = "unknown") -> None:
        """
        Snapshot cell weights into the pool (CPU, detached).

        source : "freeze"  — cell earned a freeze (long-term proven utility)
                 "prune"   — cell was elite but evicted by population pressure
                 "unknown" — legacy / untagged call
        """
        snap = {
            "w":      cell.weight.detach().cpu().clone(),
            "layer":  cell.layer_idx,
            "out":    cell.out_features,
            "in":     cell.in_features,
            "source": source,
        }
        if len(self._pool) < self.CAPACITY:
            self._pool.append(snap)
        else:
            self._pool[self._ptr % self.CAPACITY] = snap
        self._ptr += 1

    # ------------------------------------------------------------------
    # Flash  (called in _spawn for non-crossover children)
    # ------------------------------------------------------------------

    def flash(self, child: "HydraAdaptiveCell", noise: float = 0.03) -> bool:
        """
        Inject gene-pool weights into child using a randomly chosen mode.
        Returns True if any inheritance happened, False if pool was empty.
        """
        same_layer = [s for s in self._pool if s["layer"] == child.layer_idx]
        if not same_layer:
            # Fall back to any layer if nothing matches
            same_layer = list(self._pool)
        if not same_layer:
            return False

        mode = random.randint(0, 2)

        with torch.no_grad():
            if mode == self.MODE_DIRECT:
                self._flash_direct(child, same_layer, noise)
            elif mode == self.MODE_AVERAGE:
                self._flash_average(child, same_layer, noise)
            else:
                self._flash_subspace(child, same_layer, noise)
        return True

    # ── Mode implementations ───────────────────────────────────────────

    def _flash_direct(self, child, donors, noise):
        """Truncate/pad from the shape-closest donor."""
        def shape_dist(s):
            return abs(s["out"] - child.out_features) + abs(s["in"] - child.in_features)
        donor = min(donors, key=shape_dist)
        d_w   = donor["w"].to(child.weight.device)
        n_out = min(d_w.size(0), child.weight.size(0))
        n_in  = min(d_w.size(1), child.weight.size(1))
        child.weight.mul_(noise)   # small noise in uncovered regions
        child.weight[:n_out, :n_in].copy_(
            d_w[:n_out, :n_in]
            + torch.randn(n_out, n_in, device=child.weight.device) * noise
        )

    def _flash_average(self, child, donors, noise):
        """Average the overlap submatrix across all same-layer donors."""
        dev    = child.weight.device
        c_out  = child.weight.size(0)
        c_in   = child.weight.size(1)
        accum  = torch.zeros(c_out, c_in, device=dev)
        count  = torch.zeros(c_out, c_in, device=dev)
        for s in donors:
            d_w   = s["w"].to(dev)
            n_out = min(d_w.size(0), c_out)
            n_in  = min(d_w.size(1), c_in)
            accum[:n_out, :n_in] += d_w[:n_out, :n_in]
            count[:n_out, :n_in] += 1.0
        mask = count > 0
        child.weight.mul_(noise)
        child.weight[mask] = (
            accum[mask] / count[mask]
            + torch.randn_like(accum)[mask] * noise
        )

    def _flash_subspace(self, child, donors, noise):
        """
        SVD the best donor; project top-k singular vectors into child's space.

        k = max(4, min(16, min(donor_out, donor_in) // 8))

        The projection reconstructs a low-rank approximation of the donor's
        weight in the child's (possibly different) output×input space by
        clipping U and Vt to the child's dimensions. This preserves the
        principal learned directions regardless of exact shape mismatch.
        """
        def shape_dist(s):
            return abs(s["out"] - child.out_features) + abs(s["in"] - child.in_features)
        donor = min(donors, key=shape_dist)
        d_w   = donor["w"].to(child.weight.device).float()

        k = max(4, min(16, min(d_w.size(0), d_w.size(1)) // 8))
        try:
            U, S, Vt = torch.linalg.svd(d_w, full_matrices=False)
        except Exception:
            # SVD failed (degenerate matrix) — fall back to direct
            self._flash_direct(child, donors, noise)
            return

        # Clip k to what SVD actually returned
        k     = min(k, S.size(0))
        c_out = child.weight.size(0)
        c_in  = child.weight.size(1)

        # Clip singular vectors to child dimensions
        U_clip  = U[:min(U.size(0), c_out), :k]   # (c_out_clip, k)
        Vt_clip = Vt[:k, :min(Vt.size(1), c_in)]  # (k, c_in_clip)

        # Low-rank reconstruction in child's space
        recon = torch.zeros(c_out, c_in, device=child.weight.device)
        r_out = U_clip.size(0)
        r_in  = Vt_clip.size(1)
        recon[:r_out, :r_in] = (U_clip * S[:k].unsqueeze(0)) @ Vt_clip

        child.weight.copy_(
            recon + torch.randn_like(recon) * noise
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save pool to a .pt file."""
        torch.save({"pool": self._pool, "ptr": self._ptr}, path)

    def load(self, path: str) -> None:
        """Load pool from a .pt file. Silently skips if file missing."""
        if not os.path.exists(path):
            return
        data       = torch.load(path, map_location="cpu")
        self._pool = data.get("pool", [])
        self._ptr  = data.get("ptr",  0)
        print(f"  [GenePool] Loaded {len(self._pool)} snapshots from {path}")

    def __len__(self):
        return len(self._pool)



class RelativeEA:
    """Tournament-style relative EA: prune bottom cells, spawn from elite."""

    LIFETIME_DECAY         = 0.999    # decay applied on each Shapley update
    LIFETIME_UPDATE        = 0.001    # weight of new Shapley value in EMA
    TENURE_PASSIVE_DECAY   = 0.9999   # slow background decay every step
                                       # at 200 steps/interval: ~2% per interval
                                       # a cell that stops contributing loses
                                       # ~50% tenure after ~1400 steps of silence
    HEALTH_WEIGHT   = 0.70
    TENURE_WEIGHT   = 0.30
    ELITE_FRACTION  = 0.50
    SPAWN_PER_STEP  = 2

    def __init__(self, min_cells: int = 3, mutant_noise: float = 0.03,
                 min_age_to_prune: int = 0, gene_pool: "GenePool" = None):
        self.min_cells        = min_cells
        self.mutant_noise     = mutant_noise
        self.min_age_to_prune = min_age_to_prune
        self.gene_pool        = gene_pool

    @staticmethod
    def _ensure_lifetime(cell):
        if not hasattr(cell, "lifetime_contrib"):
            cell.lifetime_contrib = 0.0

    @staticmethod
    def update_lifetime(cell, sv: float):
        RelativeEA._ensure_lifetime(cell)
        cell.lifetime_contrib = (
            cell.lifetime_contrib * RelativeEA.LIFETIME_DECAY
            + sv * RelativeEA.LIFETIME_UPDATE
        )

    @staticmethod
    def composite_score(cell) -> float:
        RelativeEA._ensure_lifetime(cell)
        return (RelativeEA.HEALTH_WEIGHT * cell.health.item()
                + RelativeEA.TENURE_WEIGHT * cell.lifetime_contrib)

    def step(self, cells, adapters, make_cell_fn, make_adapter_fn,
             max_cells, n_dag_layers) -> Tuple[nn.ModuleList, nn.ModuleList, bool]:
        n = len(cells)
        for c in cells:
            self._ensure_lifetime(c)

        all_ranked = sorted(range(n),
                            key=lambda i: self.composite_score(cells[i]),
                            reverse=True)
        n_elite    = max(1, math.ceil(n * self.ELITE_FRACTION))
        elite_idx  = set(all_ranked[:n_elite])
        elite_cells = [cells[i] for i in all_ranked[:n_elite]]

        parent_tenures = sorted([getattr(c, "lifetime_contrib", 0.0)
                                  for c in elite_cells])
        inherit_tenure = (parent_tenures[len(parent_tenures) // 2]
                          if parent_tenures else 0.0) * 0.5

        # Fill phase
        if n < max_cells:
            new_cells, new_adapters = list(cells), list(adapters)
            for _ in range(self.SPAWN_PER_STEP):
                if len(new_cells) >= max_cells:
                    break
                child, adapter = self._spawn(
                    elite_cells, make_cell_fn, make_adapter_fn,
                    n_dag_layers, inherit_tenure
                )
                new_cells.append(child)
                new_adapters.append(adapter)
            return nn.ModuleList(new_cells), nn.ModuleList(new_adapters), True

        # Tournament — gradual pruning: remove only a small number of the
        # worst cells per EA call instead of all non-elite at once.
        # This prevents the boom-bust collapse where a full nuke destroys
        # all hierarchical structure in a single step.
        MAX_PRUNE_PER_STEP = 7   # hard ceiling per call
        MIN_PRUNE_PER_STEP = 1   # always remove at least 1 when over capacity

        prunable = [i for i in range(n)
                    if not cells[i].frozen
                    and int(cells[i].age.item()) >= self.min_age_to_prune
                    and i not in elite_idx]
        # Sort prunable worst-first so we cut the least useful cells first
        prunable.sort(key=lambda i: self.composite_score(cells[i]))

        # How many we're allowed to remove while staying above min_cells
        headroom  = max(0, n - self.min_cells)
        # Clamp to the gradual window
        n_prune   = min(len(prunable), headroom, MAX_PRUNE_PER_STEP)
        n_prune   = max(n_prune, MIN_PRUNE_PER_STEP if headroom >= MIN_PRUNE_PER_STEP else 0)
        if n_prune == 0:
            return cells, adapters, False

        prune_set = set(prunable[:n_prune])

        # Donate pruned cells that were in the elite set to the gene pool.
        # These are useful cells evicted only by population pressure — their
        # weights are worth preserving for future spawns.
        if self.gene_pool is not None:
            for i in prune_set:
                if i in elite_idx:
                    self.gene_pool.donate(cells[i], source="prune")

        # Per-layer floor: always keep at least one cell per layer that had
        # any cells before pruning. This prevents the total loss of
        # hierarchical structure (e.g. all L1/L2 cells wiped in one step).
        layers_present = set(
            cells[i].layer_idx for i in range(n)
            if hasattr(cells[i], "layer_idx")
        )
        for layer in layers_present:
            survivors = [i for i in range(n)
                         if i not in prune_set
                         and hasattr(cells[i], "layer_idx")
                         and cells[i].layer_idx == layer]
            if not survivors:
                # Rescue the best-scoring cell of this layer from the prune set
                in_prune = [i for i in prune_set
                            if hasattr(cells[i], "layer_idx")
                            and cells[i].layer_idx == layer]
                if in_prune:
                    rescue = max(in_prune,
                                 key=lambda i: self.composite_score(cells[i]))
                    prune_set.discard(rescue)

        new_cells    = [cells[i]    for i in range(n) if i not in prune_set]
        new_adapters = [adapters[i] for i in range(n) if i not in prune_set]

        # Tenure normalisation
        t_max = max((getattr(c, "lifetime_contrib", 0.0) for c in new_cells), default=1.0)
        if t_max > 1e-8:
            for c in new_cells:
                self._ensure_lifetime(c)
                c.lifetime_contrib /= t_max

        for _ in range(self.SPAWN_PER_STEP):
            child, adapter = self._spawn(
                elite_cells, make_cell_fn, make_adapter_fn,
                n_dag_layers, inherit_tenure
            )
            new_cells.append(child)
            new_adapters.append(adapter)

        return nn.ModuleList(new_cells), nn.ModuleList(new_adapters), True

    def _spawn(self, elite_cells, make_cell_fn, make_adapter_fn,
               n_dag_layers, inherit_tenure):
        layer_idx = random.randint(0, max(0, n_dag_layers - 1))
        child     = make_cell_fn(layer_idx)

        # Find best donor: prefer exact shape match, fall back to any same-layer
        # elite cell and adapt its weights via truncate / zero-pad so that
        # every new cell inherits at least some learned patterns.
        same_layer_exact = [c for c in elite_cells
                            if hasattr(c, "layer_idx")
                            and c.layer_idx == layer_idx
                            and c.in_features  == child.in_features
                            and c.out_features == child.out_features]
        same_layer_any   = [c for c in elite_cells
                            if hasattr(c, "layer_idx")
                            and c.layer_idx == layer_idx]

        with torch.no_grad():
            if same_layer_exact:
                # ---- Exact match: original crossover logic ----
                if len(same_layer_exact) >= 2 and random.random() < 0.3:
                    p_a, p_b = random.sample(same_layer_exact, 2)
                    mask = torch.rand_like(child.weight) > 0.5
                    child.weight.copy_(
                        torch.where(mask, p_a.weight, p_b.weight)
                        + torch.randn_like(child.weight) * self.mutant_noise
                    )
                else:
                    p = random.choice(same_layer_exact)
                    child.weight.copy_(
                        p.weight + torch.randn_like(child.weight) * self.mutant_noise
                    )
            elif same_layer_any:
                # ---- Shape mismatch: adapt donor weights ----
                # For out_features we take the overlap rows; extra child rows
                # keep kaiming init. For in_features we take the overlap cols;
                # missing child cols keep small noise from the mul_ below.
                p = random.choice(same_layer_any)
                d_w = p.weight   # (donor_out, donor_in)
                c_w = child.weight  # (child_out, child_in)

                n_out = min(d_w.size(0), c_w.size(0))
                n_in  = min(d_w.size(1), c_w.size(1))

                # Scale down uncopied regions to small noise so they're not dead
                child.weight.mul_(self.mutant_noise)
                child.weight[:n_out, :n_in].copy_(
                    d_w[:n_out, :n_in]
                    + torch.randn(n_out, n_in, device=d_w.device) * self.mutant_noise
                )
            elif self.gene_pool is not None and len(self.gene_pool) > 0:
                # ---- No live elite donor: try gene pool ----
                # This is the non-crossover path — gene pool flash with
                # random mode assignment (direct / average / subspace).
                self.gene_pool.flash(child, noise=self.mutant_noise)
            # If no donor at all, child keeps kaiming init (unchanged)

        child.lifetime_contrib = inherit_tenure
        return child, make_adapter_fn(child)


# ===========================================================================
# 5.  Transformer Encoder  (token IDs → contextual embeddings)
# ===========================================================================

class TransformerEncoder(nn.Module):
    """
    Token embedding + positional encoding + N transformer layers
    + UncertaintyGateSeq between transformer and projection head.

    Input  : (B, L) int64 token IDs
    Output : (B, L, d_model) contextual token embeddings (pre-pool)
             (B, out_dim) mean-pooled projection (fed to cell population)
    """

    def __init__(
        self,
        vocab_size:           int,
        out_dim:              int   = 128,
        d_model:              int   = 128,
        n_heads:              int   = 4,
        n_layers:             int   = 4,
        max_seq_len:          int   = 512,
        dropout:              float = 0.1,
        pad_id:               int   = 256,
        use_uncertainty_gate: bool  = True,
        gate_n_stages:        int   = 4,
        gate_n_probes:        int   = 5,
        gate_warmup_steps:    int   = 500,
    ):
        super().__init__()
        self.d_model     = d_model
        self.pad_id      = pad_id
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.uncertainty_gate: Optional[UncertaintyGateSeq] = (
            UncertaintyGateSeq(
                channels=d_model,
                n_stages=gate_n_stages,
                n_probes=gate_n_probes,
                gate_warmup_steps=gate_warmup_steps,
            ) if use_uncertainty_gate else None
        )

        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(L: int, device) -> torch.Tensor:
        """Upper-triangular mask for causal (autoregressive) attention."""
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        ids:           torch.Tensor,      # (B, L)
        global_step:   int   = 999999,
        integrity:     float = 1.0,       # from Homeostasis — gates depth
        return_hidden: bool  = False,     # if True also return (B, L, d_model)
        causal:        bool  = False,     # True → autoregressive causal mask
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns
        -------
        pooled     : (B, out_dim)  mean-pooled (bidirectional) or last-pos (causal)
        hidden     : (B, L, d_model)  or None
        token_unc  : (B, L) uncertainty per token  or None

        When causal=True the transformer uses an upper-triangular attention mask so
        position t can only attend to positions 0..t.  pooled is then the last
        non-pad hidden state rather than mean-pool, giving a proper next-token
        conditioning vector for autoregressive generation.
        """
        B, L = ids.shape
        L    = min(L, self.max_seq_len)
        ids  = ids[:, :L]
        pos  = torch.arange(L, device=ids.device).unsqueeze(0)

        pad_mask = (ids == self.pad_id)
        # Guard: if a row is fully masked, softmax(all -inf) = NaN.
        # Force position 0 unmasked so every row has at least one valid token.
        all_pad  = pad_mask.all(dim=1, keepdim=True)               # (B, 1)
        pad_mask = pad_mask & ~all_pad                             # unblock pos-0 for all-pad rows
        x = self.drop(self.token_emb(ids) + self.pos_emb(pos))   # (B, L, d_model)

        attn_mask = self._causal_mask(L, ids.device) if causal else None
        h = self.transformer(x, mask=attn_mask,
                             src_key_padding_mask=pad_mask)        # (B, L, d_model)

        token_unc: Optional[torch.Tensor] = None
        if self.uncertainty_gate is not None:
            h, token_unc = self.uncertainty_gate(
                h, global_step=global_step, integrity_scale=integrity
            )

        if causal:
            # Last non-pad position per sequence
            lengths    = (~pad_mask).sum(dim=1).clamp(min=1) - 1  # (B,)
            last_h     = h[torch.arange(B, device=h.device), lengths]  # (B, d_model)
            emb        = self.proj(last_h)
        else:
            non_pad = (~pad_mask).float().unsqueeze(-1)
            pooled  = (h * non_pad).sum(1) / non_pad.sum(1).clamp(1)
            emb     = self.proj(pooled)

        return emb, (h if return_hidden else None), token_unc


# ===========================================================================
# 6.  MLM + Causal Language Model Heads
# ===========================================================================

class MLMHead(nn.Module):
    """Masked language modelling head: (B, L, d_model) -> (B, L, vocab_size)."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class LanguageHead(nn.Module):
    """
    Autoregressive next-token projection.
    Input : (B, hidden_dim) cell population aggregate
    Output: (B, vocab_size) logits
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)



# ===========================================================================
# 6b. Loop Exit Gate
# ===========================================================================

class LoopExitGate(nn.Module):
    """
    Dynamic exit gate for looped cell inference (Oro-style).

    After each cell DAG pass the gate reads the aggregate vector and
    outputs a scalar exit probability.  We track cumulative probability
    mass so the per-loop probabilities form a proper PMF bounded to [0,1]:

        p_exit(loop k) = sigmoid(gate(h_k)) * (1 - sum(p_exit(1..k-1)))

    This means:
      - Loop 1 can exit with probability p1.
      - Loop 2 can exit with probability p2 * (1 - p1).
      - ...and so on, always summing to ≤ 1.

    At max_loops the model is forced to exit regardless.

    Entropy regularisation loss (exit_entropy_loss) penalises the model
    for always exiting at the same loop, forcing it to learn dynamic
    compute allocation rather than collapsing to a fixed depth.
    """

    def __init__(self, hidden_dim: int, max_loops: int = 4):
        super().__init__()
        self.max_loops = max_loops
        # Small MLP: aggregate → scalar exit logit
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        aggregate:    torch.Tensor,   # (B, hidden_dim) cell DAG output
        loop_idx:     int,            # current loop index (0-based)
        cumulative_p: torch.Tensor,   # (B,) probability mass already used
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        exit_prob   : (B,) probability of exiting at this loop
        new_cum_p   : (B,) updated cumulative probability mass
        log_prob    : (B,) log probability of the raw gate decision (for REINFORCE)
        """
        logit      = self.gate(aggregate).squeeze(-1)                  # (B,)
        raw_p      = torch.sigmoid(logit)                              # (B,)
        # log_prob of the raw gate decision — used for policy gradient
        # We store log(raw_p) rather than log(exit_prob) because exit_prob
        # is scaled by remaining mass which is not part of the gate's decision
        log_prob   = F.logsigmoid(logit)                               # (B,)
        remaining  = (1.0 - cumulative_p).clamp(min=0.0)
        exit_prob  = raw_p * remaining
        new_cum_p  = cumulative_p + exit_prob
        return exit_prob, new_cum_p, log_prob

    def exit_entropy_loss(
        self, exit_probs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        KL-divergence-based entropy regularisation.
        exit_probs : list of (B,) tensors, one per loop that was executed.
        Penalises low entropy over the exit distribution — forces the model
        to spread probability mass across loops rather than always exiting
        at the same one.
        """
        # Stack to (B, n_loops), normalise to a valid distribution
        stacked = torch.stack(exit_probs, dim=1).clamp(1e-6, 1.0)   # (B, K)
        stacked = stacked / stacked.sum(dim=1, keepdim=True)
        # Uniform target: each loop equally likely
        K       = stacked.size(1)
        uniform = torch.full_like(stacked, 1.0 / K)
        # KL(uniform || model) — penalises deviation from uniform exit distribution
        kl = (uniform * (uniform.log() - stacked.log())).sum(dim=1)
        return kl.mean()


# ===========================================================================
# 7.  HydraLM
# ===========================================================================

class HydraLM(nn.Module):
    """
    Full adaptive language model.

    Architecture
    ------------
    SuperBPETokenizer (external, passed in or built separately)
      -> TransformerEncoder (with UncertaintyGateSeq)
      -> Homeostasis state concat (4 scalars)
      -> DAG cell population (Shapley health, RelativeEA)
      -> LanguageHead (next-token logits)

    Pretraining objectives
    ----------------------
    1. Masked Language Modelling (MLM) on encoder hidden states.
    2. Next-token prediction (CLM) on cell population aggregate.
    Both losses are shaped by Homeostasis intrinsic rewards:
      - curiosity  : token-level loss reweighting by perplexity
      - exploration: entropy bonus on token distribution
      - consolidation: clip worst-token gradients when pain is high
      - depth_pressure: integrity passed to UncertaintyGate

    The two objectives share encoder parameters but use separate heads,
    so gradient conflict is minimal.
    """

    def __init__(
        self,
        vocab_size:          int,
        pad_id:              int   = 256,
        # Encoder
        d_model:             int   = 256,
        n_heads:             int   = 4,
        n_layers:            int   = 4,
        max_seq_len:         int   = 512,
        encoder_dim:         int   = 256,
        dropout:             float = 0.1,
        # UncertaintyGate
        gate_n_stages:       int   = 4,
        gate_n_probes:       int   = 5,
        gate_warmup_steps:   int   = 500,
        # Cell population
        initial_cells:       int   = 4,
        max_cells:           int   = 16,
        min_width:           int   = 16,    # minimum cell output width
        max_width:           int   = 512,   # maximum cell output width
        n_dag_layers:        int   = 2,
        health_decay:        float = 0.90,
        shapley_samples:     int   = 6,
        freeze_age:          int   = 800,
        freeze_health_thresh: float = 0.68,
        freeze_streak_needed: int  = 250,
        growth_interval:     int   = 200,
        min_growth_cells:    int   = 6,
        # Evolution
        min_cells:           int   = 3,
        plasticity_lr:       float = 5e-4,
        # Replay
        replay_buffer_size:  int   = 4000,
        replay_ratio:        float = 0.20,
        # Intrinsic reward weights
        w_curiosity:         float = 0.20,    # max extra weight on hard tokens
        w_exploration:       float = 1.0,     # entropy bonus multiplier
        w_consolidation:     float = 0.30,    # fraction of worst tokens to clip
        # Looped inference
        max_loops:           int   = 4,       # max cell DAG passes per token
        exit_entropy_weight: float = 0.05,    # weight of exit-gate entropy reg loss
        loop_depth_weight:   float = 0.01,    # compute tax: penalty per loop used
        reinforce_weight:    float = 0.01,    # REINFORCE policy gradient weight
        # DAG connectivity
        max_connections:     int   = 10,      # max input_sources per cell
    ):
        super().__init__()

        self.vocab_size      = vocab_size
        self.pad_id          = pad_id
        self.encoder_dim     = encoder_dim
        self.min_width       = max(1, min_width)
        self.max_width       = max(self.min_width, max_width)
        self.max_hidden_dim  = self.max_width
        self.n_dag_layers    = n_dag_layers
        self.health_decay    = health_decay
        self.freeze_age      = freeze_age
        self.freeze_health_thresh  = freeze_health_thresh
        self.freeze_streak_needed  = freeze_streak_needed
        self.growth_interval = growth_interval
        self.min_growth_cells = min_growth_cells
        self.plasticity_lr   = plasticity_lr
        self.replay_ratio    = replay_ratio
        self.max_cells       = max_cells

        # Intrinsic reward weights
        self.w_curiosity          = w_curiosity
        self.w_exploration        = w_exploration
        self.w_consolidation      = w_consolidation
        # Looped inference
        self.max_loops            = max_loops
        self.exit_entropy_weight  = exit_entropy_weight
        self.loop_depth_weight    = loop_depth_weight
        self.reinforce_weight     = reinforce_weight
        # DAG connectivity
        self.max_connections      = max_connections

        # --- Transformer encoder ---
        self.encoder = TransformerEncoder(
            vocab_size           = vocab_size,
            out_dim              = encoder_dim,
            d_model              = d_model,
            n_heads              = n_heads,
            n_layers             = n_layers,
            max_seq_len          = max_seq_len,
            dropout              = dropout,
            pad_id               = pad_id,
            use_uncertainty_gate = True,
            gate_n_stages        = gate_n_stages,
            gate_n_probes        = gate_n_probes,
            gate_warmup_steps    = gate_warmup_steps,
        )

        # --- MLM head (operates on raw transformer hidden states) ---
        self.mlm_head = MLMHead(d_model, vocab_size)

        # --- Homeostasis (4-dim state appended to encoder output) ---
        # warmup_steps matched to gate_warmup_steps so pain/integrity don't
        # fire until the model has had time to stabilise
        self.homeostasis    = Homeostasis(warmup_steps=max(gate_warmup_steps, 500))
        self._homeo_dim     = 4

        # Cell input dimensions:
        # L0: encoder_dim + homeo_dim
        # Lk: encoder_dim + homeo_dim + encoder_dim (prev layer aggregate slice)
        l0_in = encoder_dim + self._homeo_dim
        lk_in = l0_in + encoder_dim
        self._layer_in_dims = [l0_in if k == 0 else lk_in
                                for k in range(n_dag_layers)]

        # --- Cell population ---
        self.cells = nn.ModuleList()
        for i in range(initial_cells):
            self.cells.append(self._make_cell(0 if i < max(1, initial_cells - 1) else 1))
        self.cell_adapters = nn.ModuleList([self._make_adapter(c) for c in self.cells])

        # --- Language head ---
        self.language_head = LanguageHead(self.max_hidden_dim, vocab_size)

        # --- Loop exit gate ---
        self.exit_gate = LoopExitGate(self.max_hidden_dim, max_loops=max_loops)

        # --- Loop input projection ---
        # Projects (enc + loop_ctx) down to base_in for root cells on loops 2+.
        # enc is (encoder_dim + homeo_dim), loop_ctx adds max_hidden_dim.
        # Registered as a proper module so it's in state_dict, DNA, and optimizer.
        loop_proj_in = (encoder_dim + self._homeo_dim) + self.max_width
        loop_proj_out = encoder_dim + self._homeo_dim
        self._loop_proj = nn.Linear(loop_proj_in, loop_proj_out, bias=False)

        # --- Support systems ---
        self.shapley   = ShapleyAttributor(n_samples=shapley_samples)
        self.gene_pool = GenePool()
        self.rel_ea    = RelativeEA(
            min_cells        = min_cells,
            min_age_to_prune = growth_interval * 2,
            gene_pool        = self.gene_pool,
        )
        self.replay_buf = _ReservoirBuffer(capacity=replay_buffer_size)

        self.register_buffer("error_ema",       torch.tensor(1.0))
        self.register_buffer("long_error_ema",  torch.tensor(1.0))
        self.register_buffer("step_count",      torch.tensor(0))
        self.register_buffer("degraded_streak", torch.tensor(0.0))
        self.register_buffer("_prev_loss",      torch.tensor(100.0))
        self.register_buffer("_prev_clm_loss",  torch.tensor(100.0))

    # ------------------------------------------------------------------
    # Cell factory
    # ------------------------------------------------------------------

    def _make_cell(self, layer_idx: int = 0,
                   input_sources: list = None) -> HydraAdaptiveCell:
        w             = random.randint(self.min_width, self.max_width)
        layer_idx     = min(layer_idx, self.n_dag_layers - 1)
        input_sources = list(input_sources) if input_sources is not None else []
        # Enforce connection cap — trim to max_connections if over limit
        if len(input_sources) > self.max_connections:
            input_sources = random.sample(input_sources, self.max_connections)
        # Root cells receive encoder+homeo only.
        # Each explicit source adds max_hidden_dim to the input.
        base_in = self.encoder_dim + self._homeo_dim
        in_dim  = base_in + len(input_sources) * self.max_hidden_dim
        cell    = HydraAdaptiveCell(
            in_features   = in_dim,
            out_features  = w,
            plasticity_lr = self.plasticity_lr,
            health_decay  = self.health_decay,
            layer_idx     = layer_idx,
            input_sources = input_sources,
        )
        # New cells start with low health so the health-gated aggregate
        # isn't immediately diluted. They earn weight as Shapley scores accumulate.
        with torch.no_grad():
            cell.health.fill_(0.1)
        device = next(self.parameters()).device
        return cell.to(device)

    def _make_adapter(self, cell) -> nn.Linear:
        device = next(self.parameters()).device
        return nn.Linear(cell.out_features, self.max_hidden_dim).to(device)

    # ------------------------------------------------------------------
    # DAG topology helpers
    # ------------------------------------------------------------------

    def _topo_order(self) -> List[int]:
        """Kahn's algorithm topological sort over the cell DAG.
        Falls back to index order if a cycle is detected (shouldn't happen,
        but guards against malformed input_sources after pruning)."""
        n      = len(self.cells)
        in_deg = [0] * n
        adj    = [[] for _ in range(n)]
        for i, cell in enumerate(self.cells):
            for src in cell.input_sources:
                if 0 <= src < n:
                    adj[src].append(i)
                    in_deg[i] += 1
        queue = [i for i in range(n) if in_deg[i] == 0]
        order: List[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
        if len(order) != n:
            return list(range(n))   # cycle fallback
        return order

    def _dag_input(self, idx: int, enc: torch.Tensor,
                   cell_outs: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Build the input tensor for cell idx from enc + its source outputs."""
        sources = self.cells[idx].input_sources
        if not sources:
            return enc
        src_outs = [
            cell_outs.get(s, torch.zeros(enc.size(0), self.max_hidden_dim,
                                          device=enc.device))
            for s in sources
        ]
        return torch.cat([enc] + src_outs, dim=1)

    def _update_depths(self) -> None:
        """Recompute layer_idx for every cell from topological depth."""
        n     = len(self.cells)
        depth = [0] * n
        for i in self._topo_order():
            srcs      = self.cells[i].input_sources
            depth[i]  = (max(depth[s] for s in srcs if s < n) + 1
                         if srcs else 0)
            self.cells[i].layer_idx = min(depth[i], self.n_dag_layers - 1)

    def _prune_dead_connections(self) -> None:
        """Remove any input_sources references to cells that no longer exist."""
        valid = set(range(len(self.cells)))
        for cell in self.cells:
            cell.input_sources = [s for s in cell.input_sources if s in valid]

    # ------------------------------------------------------------------
    # Forward (inference / supervised)
    # ------------------------------------------------------------------

    def forward(
        self,
        ids:         torch.Tensor,    # (B, L) int64
        plastic:     bool = False,
        global_step: int  = None,
        causal:      bool = True,     # True for generation; False for MLM training
    ) -> torch.Tensor:
        """
        Returns (B, vocab_size) logits from language head.

        causal=True  (default, generation):
            Uses causal attention mask + last-position hidden state.
            The model cannot peek at future tokens — proper AR generation.
        causal=False (MLM pretraining bidirectional encoder pass):
            Uses full bidirectional attention + mean-pool.
        """
        if global_step is None:
            global_step = int(self.step_count.item())

        integrity = self.homeostasis.integrity.item()
        enc, _, _ = self.encoder(ids, global_step=global_step,
                                  integrity=integrity, causal=causal)

        B = enc.size(0)
        h_state = self.homeostasis.get_vector(B, enc.device)
        fused   = torch.cat([enc, h_state], dim=1)         # (B, encoder_dim + 4)

        aggregated, _, _, _, _ = self._cell_forward_looped(
            fused, plastic=plastic, training_mode=self.training
        )
        return self.language_head(aggregated)

    def _cell_forward(
        self, enc: torch.Tensor, plastic: bool = False
    ) -> torch.Tensor:
        """
        Run the cell DAG in topological order.
        Each cell receives enc + the adapter-projected outputs of its
        explicit input_sources (root cells receive enc only).
        Returns health-gated aggregate over all cell outputs.
        """
        order:     List[int]                      = self._topo_order()
        cell_outs: Dict[int, torch.Tensor]        = {}
        all_outs:  List[torch.Tensor]             = []

        for idx in order:
            x_in = self._dag_input(idx, enc, cell_outs)
            out  = self.cell_adapters[idx](
                       self.cells[idx](x_in, plastic=plastic))
            cell_outs[idx] = out
            all_outs.append(out)

        if not all_outs:
            raise RuntimeError("No cell outputs produced.")

        device        = all_outs[0].device
        health_scores = torch.stack([c.health.to(device) for c in self.cells])
        gates         = F.softmax(health_scores.float(), dim=0)
        stacked       = torch.stack(all_outs, dim=0)                # (N, B, H)
        return (stacked * gates.view(-1, 1, 1)).sum(0)              # (B, H)

    def _cell_forward_looped(
        self,
        enc:     torch.Tensor,   # (B, encoder_dim + homeo_dim) fused encoder+homeo
        plastic: bool = False,
        training_mode: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """
        Looped cell DAG forward pass (Oro-style latent looping).

        Each iteration runs the full DAG then asks the exit gate whether to
        stop.  The aggregate from the previous loop is concatenated back onto
        `enc` as a residual conditioning signal so cells on the next pass can
        refine rather than repeat.

        The loop conditioning vector is zero on the first pass (no prior
        aggregate), so root cells on loop 0 behave exactly as before.

        Returns
        -------
        final_agg   : (B, max_hidden_dim)  output of the exit loop
        exit_probs  : list of (B,) per-loop exit probabilities (for entropy reg)
        n_loops_run : how many loops were actually executed
        """
        B          = enc.size(0)
        device     = enc.device
        cumulative = torch.zeros(B, device=device)
        exit_probs: List[torch.Tensor] = []

        # Loop conditioning: zero on first pass, previous aggregate thereafter.
        # Concatenated to enc so cells see both the encoder representation and
        # what the DAG concluded on the previous iteration.
        loop_ctx   = torch.zeros(B, self.max_hidden_dim, device=device)
        final_agg  = loop_ctx.clone()

        loop_log_probs: List[torch.Tensor] = []   # gate log probs for REINFORCE
        loop_aggs:      List[torch.Tensor] = []   # per-loop aggregates for rewards

        for loop_idx in range(self.max_loops):
            # Combine encoder embedding with previous loop output
            enc_in = torch.cat([enc, loop_ctx], dim=1)   # (B, base_in + max_hidden_dim)

            agg = self._cell_forward_with_enc(enc_in, plastic=plastic)

            # Exit gate — now also returns log_prob for REINFORCE
            exit_p, cumulative, log_prob = self.exit_gate(agg, loop_idx, cumulative)
            exit_probs.append(exit_p)
            loop_log_probs.append(log_prob)
            loop_aggs.append(agg)
            final_agg  = agg
            loop_ctx   = agg

            if not training_mode:
                if cumulative.mean().item() > 0.5 or loop_idx == self.max_loops - 1:
                    break

        return final_agg, exit_probs, loop_idx + 1, loop_log_probs, loop_aggs

    def _cell_forward_with_enc(
        self, enc: torch.Tensor, plastic: bool = False
    ) -> torch.Tensor:
        """
        Identical to _cell_forward but accepts a variable-width enc tensor.
        The topo-DAG _dag_input uses enc as the base for root cells, so as
        long as enc width matches the cell's in_features this works directly.
        Root cells were built with in_dim = encoder_dim + homeo_dim, but on
        loop passes enc is wider (+ max_hidden_dim). We project it back down
        to the expected base width before passing to cells that don't have
        explicit sources, so their weight shapes stay correct.
        """
        # Project loop-augmented enc down to base_in for root cells.
        # _loop_proj is registered in __init__ — always available, always on
        # the right device, always in state_dict and optimizer.
        base_in  = self.encoder_dim + self._homeo_dim
        if enc.size(1) > base_in:
            enc_root = self._loop_proj(enc)
        else:
            enc_root = enc

        order:     List[int]               = self._topo_order()
        cell_outs: Dict[int, torch.Tensor] = {}
        all_outs:  List[torch.Tensor]      = []

        for idx in order:
            sources = self.cells[idx].input_sources
            if not sources:
                x_in = enc_root
            else:
                src_outs = [
                    cell_outs.get(s, torch.zeros(enc.size(0), self.max_hidden_dim,
                                                  device=enc.device))
                    for s in sources
                ]
                x_in = torch.cat([enc_root] + src_outs, dim=1)

            out = self.cell_adapters[idx](
                      self.cells[idx](x_in, plastic=plastic))
            cell_outs[idx] = out
            all_outs.append(out)

        if not all_outs:
            raise RuntimeError("No cell outputs produced.")

        device        = all_outs[0].device
        health_scores = torch.stack([c.health.to(device) for c in self.cells])
        gates         = F.softmax(health_scores.float(), dim=0)
        stacked       = torch.stack(all_outs, dim=0)
        return (stacked * gates.view(-1, 1, 1)).sum(0)

    # ------------------------------------------------------------------
    # Pretraining step — MLM + CLM + intrinsic rewards
    # ------------------------------------------------------------------

    def pretrain_step(
        self,
        ids:          torch.Tensor,             # (B, L) raw token IDs
        optimizer:    torch.optim.Optimizer,    # main optimizer (all params)
        tokenizer:    SuperBPETokenizer,
        mlm_weight:   float = 0.5,              # weight of MLM vs CLM loss
        unc_reg_weight: float = 0.01,
        pg_optimizer: Optional[torch.optim.Optimizer] = None,  # exit gate only
    ) -> Tuple[float, Dict[str, float]]:
        """
        Combined MLM + CLM pretraining step with homeostasis intrinsic rewards.

        Intrinsic reward application
        ----------------------------
        curiosity_bonus (energy-based):
          Token-level loss reweighting. We compute per-token CE loss and
          upweight tokens with high loss (hard / surprising tokens) by a
          factor proportional to (1 - energy). When the model is "tired"
          (low energy), it leans harder into the tokens it finds difficult.

        exploration_bonus (excitement-based):
          Adds H(p) entropy regularisation to the CLM logits to encourage
          a more diverse next-token distribution when the model is excited
          by novelty. Prevents probability mass collapsing to a single token.

        consolidation_clip (pain-based):
          When pain is high the model is struggling. We clip the loss
          contribution of the hardest `consolidation_clip * 100`% of tokens
          to their median value, preventing a few outlier tokens from
          destabilising the gradient entirely.

        depth_pressure (integrity-based):
          Passed as `integrity` to the encoder which forwards it to
          UncertaintyGateSeq. Low integrity → thresholds shift down →
          more tokens route through deeper computation stages.

        Returns (total_loss_value, reward_dict).
        """
        self.train()
        optimizer.zero_grad()

        global_step = int(self.step_count.item())
        rewards     = self.homeostasis.intrinsic_rewards()
        integrity   = rewards["depth_pressure"]  # passed to encoder

        # ---- Replay buffer: store current batch ----------------------
        # Always add the incoming batch to the reservoir.
        self.replay_buf.add_batch(ids.cpu())

        # Homeostasis-gated trajectory selection:
        # Mix in replayed sequences when pain is elevated (model struggling)
        # or excitement is high (novelty detected) — replay the most
        # informative past trajectories at exactly those moments.
        # replay_ratio controls the base fraction; pain/excitement scale it.
        pain_mag    = abs(self.homeostasis.pain.item())       # [0, 1]
        excitement  = self.homeostasis.excitement.item()      # [0, 1]
        # Gate: replay more when the model is either struggling or excited
        homeo_gate  = min(1.0, pain_mag * 2.0 + excitement * 0.5)
        eff_ratio   = self.replay_ratio * (1.0 + homeo_gate)  # up to 2x base ratio
        n_replay    = int(ids.size(0) * eff_ratio)

        if n_replay > 0 and len(self.replay_buf) >= n_replay * 2:
            replay_batch = self.replay_buf.sample(n_replay, device=ids.device)
            if replay_batch is not None:
                # Pad/trim replay batch to match current sequence length
                L_cur = ids.size(1)
                L_rep = replay_batch.size(1)
                if L_rep < L_cur:
                    pad = torch.full(
                        (replay_batch.size(0), L_cur - L_rep),
                        tokenizer.PAD_ID, dtype=torch.long, device=ids.device
                    )
                    replay_batch = torch.cat([replay_batch, pad], dim=1)
                else:
                    replay_batch = replay_batch[:, :L_cur]
                ids = torch.cat([ids, replay_batch], dim=0)

        # ---- MLM objective ----------------------------------------
        masked_ids, mlm_targets = tokenizer.apply_mask(ids)

        enc, hidden, token_unc = self.encoder(
            masked_ids,
            global_step   = global_step,
            integrity     = integrity,
            return_hidden = True,
        )

        mlm_logits     = self.mlm_head(hidden)                      # (B, L, V)
        B, L, V        = mlm_logits.shape

        # Per-token MLM loss (unreduced)
        mlm_loss_raw   = F.cross_entropy(
            mlm_logits.reshape(B * L, V),
            mlm_targets.reshape(B * L),
            ignore_index = -100,
            reduction    = "none",
        ).reshape(B, L)                                              # (B, L)

        # --- Curiosity: reweight by per-token difficulty ----------------
        # When energy is low the model upweights high-loss tokens.
        # weight_i = 1 + w_curiosity * (1 - energy) * norm_loss_i
        # where norm_loss_i is the token's loss normalised to [0,1] per batch.
        curiosity_strength = rewards["curiosity"] * self.w_curiosity
        if curiosity_strength > 1e-4:
            valid_mask  = (mlm_targets != -100).float()
            token_loss_clipped = mlm_loss_raw.detach() * valid_mask
            loss_max    = token_loss_clipped.max().clamp(min=1e-8)
            norm_loss   = token_loss_clipped / loss_max              # [0, 1]
            curiosity_w = 1.0 + curiosity_strength * norm_loss
            mlm_loss_raw = mlm_loss_raw * curiosity_w

        # --- Consolidation: clip worst-token gradients on SEVERE pain only --
        # Consolidation is an emergency brake for catastrophic instability,
        # not a routine cushion for small regressions.
        #
        # Mild pain (small step regression) should leave gradients alone —
        # the model needs full gradient signal to correct itself.
        # Only clip when pain is high AND loss is well above the long-run
        # baseline (ratio > 1.15), indicating genuine destabilisation.
        #
        # clip_ceil sentinel: inf = no clipping.
        consolidation_frac = rewards["consolidation"]
        clip_ceil: float   = float("inf")
        severe_pain = (
            consolidation_frac > 0.15           # pain must be meaningful
            and self.homeostasis.clm_loss_ema.item() > 0
            and self._prev_clm_loss.item() / (self.homeostasis.clm_loss_ema.item() + 1e-8) > 1.15
        )
        if severe_pain:
            valid_mask = (mlm_targets != -100).float()
            flat_valid = (mlm_loss_raw.detach() * valid_mask).reshape(-1)
            n_valid    = valid_mask.sum().int().item()
            if n_valid > 4:
                k          = max(1, int(n_valid * consolidation_frac))
                top_vals,_ = torch.topk(flat_valid[flat_valid > 0], k)
                clip_ceil  = top_vals.median().item()
                mlm_loss_raw = mlm_loss_raw.clamp(max=clip_ceil)

        valid_mlm = mlm_loss_raw[mlm_targets != -100]
        mlm_loss  = valid_mlm.mean() if valid_mlm.numel() > 0 else mlm_loss_raw.mean() * 0.0

        # ---- CLM objective (next-token) --------------------------------
        # Shift: predict token[t] from prefix[0:t]
        # Use the MLM-encoder hidden states for efficiency (same forward pass)
        # CLM loss over non-pad positions
        if L > 1:
            clm_logits  = self.mlm_head(hidden[:, :-1, :])          # (B, L-1, V)
            clm_targets = ids[:, 1:]                                 # (B, L-1)
            clm_mask    = (clm_targets != self.pad_id)

            clm_loss_raw = F.cross_entropy(
                clm_logits.reshape(B * (L - 1), V),
                clm_targets.reshape(B * (L - 1)),
                ignore_index = self.pad_id,
                reduction    = "none",
            ).reshape(B, L - 1)

            # Curiosity reweighting on CLM too
            if curiosity_strength > 1e-4:
                valid_m     = clm_mask.float()
                clm_clipped = clm_loss_raw.detach() * valid_m
                norm_clm    = clm_clipped / (clm_clipped.max().clamp(min=1e-8))
                curiosity_w = 1.0 + curiosity_strength * norm_clm
                clm_loss_raw = clm_loss_raw * curiosity_w

            # Consolidation clip on CLM — only if severe pain triggered above
            if severe_pain and clip_ceil < float("inf"):
                clm_loss_raw = clm_loss_raw.clamp(max=clip_ceil)

            valid_clm = clm_loss_raw[clm_mask]
            clm_loss  = valid_clm.mean() if valid_clm.numel() > 0 else clm_loss_raw.mean() * 0.0
        else:
            clm_loss = torch.tensor(0.0, device=ids.device)

        # ---- Cell population CLM head (looped) ----------------------------
        # Re-encode causally then run the looped cell DAG.
        # exit_probs is used for entropy regularisation (anti-collapse).
        # n_loops feeds into homeostasis excitement signal.
        B2 = ids.size(0)
        enc_causal, _, _ = self.encoder(
            ids, global_step=global_step, integrity=integrity, causal=True
        )
        h_state_c   = self.homeostasis.get_vector(B2, enc_causal.device)
        fused_c     = torch.cat([enc_causal, h_state_c], dim=1)
        aggregated, exit_probs, n_loops_run, loop_log_probs, loop_aggs =             self._cell_forward_looped(fused_c, training_mode=True)
        cell_logits = self.language_head(aggregated)               # (B, V)

        # Target: next token after the last non-pad position in each sequence.
        # ids is (B, L); shift by 1 so we predict ids[:, last_pos+1].
        pad_mask_ids = (ids == self.pad_id)
        # lengths = number of non-pad tokens. Next token is at position `lengths`
        # (0-indexed), which is 1 past the last real token.
        lengths  = (~pad_mask_ids).sum(dim=1)                       # (B,) in [0, L]
        L_ids    = ids.size(1)
        next_pos = lengths.clamp(max=L_ids - 1)                     # safe index
        cell_targets = ids[torch.arange(B2, device=ids.device), next_pos]
        # Mask sequences where there is no room for a next token (fully packed or all-pad)
        no_next  = (lengths == 0) | (lengths >= L_ids)
        cell_targets = cell_targets.masked_fill(no_next, -100)
        cell_targets = cell_targets.masked_fill(cell_targets == self.pad_id, -100)
        has_valid_cell = (cell_targets != -100).any()
        if has_valid_cell:
            cell_loss = F.cross_entropy(cell_logits, cell_targets, ignore_index=-100)
        else:
            cell_loss = cell_logits.sum() * 0.0   # zero loss, keeps graph alive

        # ---- Exploration bonus (entropy regularisation) ----------------
        exploration_w = rewards["exploration"] * self.w_exploration
        if exploration_w > 1e-6:
            probs_cell   = F.softmax(cell_logits, dim=-1).clamp(1e-8, 1.0)
            entropy_cell = -(probs_cell * probs_cell.log()).sum(-1).mean()
            cell_loss    = cell_loss - exploration_w * entropy_cell

        # ---- Uncertainty regularisation --------------------------------
        unc_reg = torch.tensor(0.0, device=ids.device)
        if token_unc is not None and unc_reg_weight > 0:
            unc_reg = self.encoder.uncertainty_gate.uncertainty_reg_loss(
                token_unc.reshape(-1)
            )

        # ---- Exit gate entropy regularisation --------------------------
        exit_entropy_loss = torch.tensor(0.0, device=ids.device)
        if len(exit_probs) > 1 and self.exit_entropy_weight > 0:
            exit_entropy_loss = self.exit_gate.exit_entropy_loss(exit_probs)
        loop_excitement = (n_loops_run - 1) / max(self.max_loops - 1, 1)

        # ---- Compute tax (metabolic cost of thinking) -------------------
        compute_tax = (
            torch.tensor(float(n_loops_run) / self.max_loops, device=ids.device)
            * self.loop_depth_weight
        )

        # ---- REINFORCE policy gradient on exit gate -------------------
        # The exit gate decides how many loops to run. We treat each loop's
        # exit decision as an action and reward it based on how much that
        # loop improved the CLM loss.
        #
        # Reward signal: per-loop CLM improvement
        #   reward[k] = clm_loss(loop k-1 agg) - clm_loss(loop k agg)
        #   Positive = loop k helped. Negative = loop k hurt or didn't help.
        #
        # REINFORCE update (second backward pass, exit gate params only):
        #   loss_pg = -sum_k( log_prob[k] * advantage[k] )
        #
        # Advantages are normalised across the batch to reduce variance.
        # Only runs when reinforce_weight > 0 and we have multiple loops.
        # ---- Compute REINFORCE rewards (no graph yet — all detached) ----
        # Rewards are computed purely from values, no autograd needed.
        # The actual log_prob graph is built AFTER optimizer.step() on fresh weights.
        reinforce_loss  = torch.tensor(0.0, device=ids.device)
        pg_rewards_t    = None
        pg_loop_aggs    = None
        do_reinforce    = (
            pg_optimizer is not None
            and self.reinforce_weight > 0
            and len(loop_aggs) > 1
        )
        if do_reinforce:
            with torch.no_grad():
                loop_clm_losses = []
                for agg in loop_aggs:
                    lgt = self.language_head(agg.detach())
                    lc  = (F.cross_entropy(lgt, cell_targets, ignore_index=-100)
                           if has_valid_cell
                           else torch.tensor(0.0, device=ids.device))
                    loop_clm_losses.append(lc.item())

                baseline_loss = loop_clm_losses[0] + 0.1
                rewards_list  = []
                prev = baseline_loss
                for lc in loop_clm_losses:
                    rewards_list.append(prev - lc)
                    prev = lc
                rewards_t = torch.tensor(rewards_list, device=ids.device)
                if rewards_t.std() > 1e-6:
                    rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
                pg_rewards_t = rewards_t
                # Store detached aggs for fresh gate forward after supervised step
                pg_loop_aggs = [a.detach().clone() for a in loop_aggs]

        # ---- Combined supervised loss ----------------------------------
        total_loss = (mlm_weight * mlm_loss
                      + (1.0 - mlm_weight) * clm_loss
                      + 0.3 * cell_loss
                      + unc_reg_weight * unc_reg
                      + self.exit_entropy_weight * exit_entropy_loss
                      + compute_tax)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            parts = {"mlm": mlm_loss, "clm": clm_loss,
                     "cell": cell_loss, "unc_reg": unc_reg}
            bad = [k for k, v in parts.items()
                   if torch.isnan(v) or torch.isinf(v)]
            print(f"  [NaN] skipping step — bad components: {bad or 'total only'}")
            optimizer.zero_grad()
            loss_val = float("nan")
        else:
            # ---- Pass 1: Supervised — all params except exit gate ----
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            loss_val = total_loss.item()

            # ---- Pass 2: REINFORCE — exit gate only ----
            # Build a FRESH graph through the exit gate on its now-updated weights.
            # pg_loop_aggs are fully detached clones — no stale graph connections.
            # This is the only reliable way: build the RL graph after the
            # supervised step so the weights are at their final version.
            if do_reinforce and pg_loop_aggs is not None:
                pg_optimizer.zero_grad()
                fresh_log_probs = []
                cum = torch.zeros(pg_loop_aggs[0].size(0),
                                  device=pg_loop_aggs[0].device)
                for k, agg in enumerate(pg_loop_aggs):
                    _, new_cum, lp = self.exit_gate(agg, k, cum)
                    fresh_log_probs.append(lp)
                    cum = new_cum.detach()
                log_probs_t    = torch.stack(fresh_log_probs, dim=0)
                advantages     = pg_rewards_t.unsqueeze(1).expand_as(log_probs_t)
                reinforce_loss = -(log_probs_t * advantages).mean()
                if not (torch.isnan(reinforce_loss) or torch.isinf(reinforce_loss)):
                    pg_loss = reinforce_loss * self.reinforce_weight
                    pg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.exit_gate.parameters(), max_norm=0.5
                    )
                    pg_optimizer.step()

        # ---- Homeostasis update ----------------------------------------
        # Feed CLM loss only — decoupled from composite training loss.
        # Homeostasis responds to language learning quality, not regularisation.
        clm_loss_val     = clm_loss.item() if not math.isnan(loss_val) else self._prev_clm_loss.item()
        prev_clm_loss    = self._prev_clm_loss.item()
        self.homeostasis.update(clm_loss_val, prev_clm_loss)
        # Loop excitement still bumps excitement — more loops = harder input
        if loop_excitement > 0.0:
            with torch.no_grad():
                bump = loop_excitement * 0.05
                self.homeostasis.excitement = torch.tensor(
                    min(1.0, self.homeostasis.excitement.item() + bump)
                )
        self._prev_loss.fill_(loss_val)
        self._prev_clm_loss.fill_(clm_loss_val)

        # ---- Evolution -------------------------------------------------
        _, sv_info = self._evolve(loss_val, fused_c.detach(), cell_targets)

        reward_info = {
            **rewards,
            **sv_info,          # sv_cell_0 … sv_cell_N for StatsTracker
            "mlm_loss":         mlm_loss.item(),
            "clm_loss":         clm_loss.item(),
            "cell_loss":        cell_loss.item(),
            "exit_entropy":     exit_entropy_loss.item(),
            "compute_tax":      compute_tax.item(),
            "reinforce":        reinforce_loss.item(),
            "n_loops":          n_loops_run,
            "replay_buf_size":  len(self.replay_buf),
            "total_loss":       loss_val,
        }
        return loss_val, reward_info

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def _evolve(self, loss_val: float, enc: torch.Tensor,
                y: torch.Tensor) -> bool:
        changed = False
        self.step_count += 1

        alpha = 0.10
        self.error_ema      = (1 - alpha) * self.error_ema + alpha * loss_val
        self.long_error_ema = (1 - alpha * 0.1) * self.long_error_ema + alpha * 0.1 * loss_val

        # Passive tenure decay — applied every step regardless of Shapley.
        # Cells that stop contributing slowly lose their tournament advantage.
        # Active cells recover tenure via Shapley updates faster than they lose it.
        for cell in self.cells:
            RelativeEA._ensure_lifetime(cell)
            cell.lifetime_contrib *= RelativeEA.TENURE_PASSIVE_DECAY

        # Shapley health attribution — full DAG, all cells
        # Path-aware: downstream cells propagate credit to their upstream sources.
        sv_info: Dict[str, float] = {}
        if self.cells:
            sv = self.shapley.compute(
                list(self.cells), list(self.cell_adapters), enc, y,
                self.language_head, self.max_hidden_dim, self.pad_id
            )
            sv_max = max(sv) + 1e-8
            for cell_idx, cell in enumerate(self.cells):
                norm_sv = sv[cell_idx] / sv_max
                new_h   = (cell.health.item() * self.health_decay
                           + norm_sv * (1.0 - self.health_decay))
                RelativeEA.update_lifetime(cell, norm_sv)
                earned  = cell.tick_health_streak(
                    new_h, self.freeze_health_thresh, self.freeze_streak_needed
                )
                if (not cell.frozen and earned
                        and cell.age.item() >= self.freeze_age):
                    cell.freeze()
                    self.gene_pool.donate(cell, source="freeze")
                    print(f"  [Freeze] Cell {cell_idx} "
                          f"layer={cell.layer_idx} age={int(cell.age.item())} -> FROZEN "
                          f"[donated to gene pool, size={len(self.gene_pool)}]")
                # Expose per-cell Shapley values so StatsTracker can log them
                sv_info[f"sv_cell_{cell_idx}"] = norm_sv

        # Conditional unfreeze (sustained degradation)
        degrading = self.error_ema.item() > self.long_error_ema.item() * 1.05
        if degrading:
            self.degraded_streak += 1.0
        else:
            self.degraded_streak.fill_(0.0)
        if self.degraded_streak.item() >= 5:
            frozen_cells = [(i, c) for i, c in enumerate(self.cells) if c.frozen]
            if frozen_cells:
                _, oldest = max(frozen_cells, key=lambda ic: ic[1].age.item())
                oldest.unfreeze()
                self.degraded_streak.fill_(0.0)
                print("  [Unfreeze] Cell re-enters gradient descent")
                changed = True

        # Relative EA (every growth_interval steps)
        if self.step_count.item() % self.growth_interval == 0:
            new_cells, new_adapters, ea_changed = self.rel_ea.step(
                self.cells, self.cell_adapters,
                self._make_cell, self._make_adapter,
                self.max_cells, self.n_dag_layers,
            )
            if ea_changed:
                self.cells         = new_cells
                self.cell_adapters = new_adapters
                changed = True

            # Growth trigger
            n = len(self.cells)
            underpop     = n < self.min_growth_cells
            loss_high    = loss_val > self.error_ema.item()
            if n < self.max_cells and (underpop or loss_high):
                self._grow()
                changed = True

        # After any structural change, recompute DAG depths and remove
        # stale input_source references that point to pruned cells.
        if changed:
            self._prune_dead_connections()
            self._update_depths()

        return changed, sv_info

    def _grow(self):
        """
        Spawn one new cell, wiring it into the DAG.

        Source selection
        ----------------
        Bias toward 1-2 explicit sources (making the new cell a consumer of
        existing cells) but occasionally produce a root cell (no sources).
        The new cell's in_dim is derived from its source list so the weight
        shape is always correct.  Warm-starting from same-layer donors is
        attempted only when out_features match.
        """
        n       = len(self.cells)
        new_idx = n

        # Choose how many sources — bias toward 1-3, cap at max_connections
        max_src = min(self.max_connections, n)
        # Weight distribution: root(1), 1-src(4), 2-src(4), 3-src(3), 4+(1 each)
        # Falls off after 3 to discourage overly wide fan-in
        w = [1] + [4, 4, 3] + [1] * max(0, max_src - 3)
        w = w[:max_src + 1]
        n_src   = random.choices(range(max_src + 1), weights=w)[0] if n > 0 else 0
        sources = random.sample(range(n), min(n_src, n)) if n_src > 0 else []

        # Depth = one layer deeper than deepest source
        depth = (max(self.cells[s].layer_idx for s in sources) + 1
                 if sources else 0)

        child   = self._make_cell(layer_idx=depth, input_sources=sources)
        # Only warm-start from donors with exactly matching weight shape
        donors  = [c for c in self.cells
                   if c.layer_idx == depth
                   and not c.frozen
                   and c.in_features  == child.in_features
                   and c.out_features == child.out_features]
        if len(donors) >= 2:
            p_a, p_b = random.sample(donors, 2)
            with torch.no_grad():
                mask = torch.rand_like(child.weight) > 0.5
                child.weight.copy_(
                    torch.where(mask, p_a.weight, p_b.weight)
                    + torch.randn_like(child.weight) * 0.02
                )
        elif donors:
            p = donors[0]
            with torch.no_grad():
                child.weight.copy_(p.weight + torch.randn_like(child.weight) * 0.02)

        adapter = self._make_adapter(child)
        self.cells.append(child)
        self.cell_adapters.append(adapter)
        src_str = str(sources) if sources else "root"
        print(f"  [Grow] Cell {new_idx} L{depth} w={child.out_features} "
              f"src={src_str} pop={len(self.cells)}")

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:       torch.Tensor,   # (B, S) int64 seed
        max_new:          int   = 128,
        temperature:      float = 1.0,
        top_k:            int   = 50,
        top_p:            float = 0.95,
        eos_id:           int   = 259,    # SuperBPETokenizer.EOS_ID
        suppress_control: bool  = True,   # mask non-printable byte tokens
    ) -> torch.Tensor:
        """
        Autoregressive generation through the full model:
          ctx → encoder (causal) → last-pos emb → homeo concat → cell DAG → language_head

        The evolutionary cell population is the learned predictor.
        The encoder (causal) gives it a properly masked prefix representation.
        language_head projects the cell aggregate to vocab logits.

        suppress_control=True masks byte IDs 0-8 and 11-31 (non-printable
        control characters) which are valid vocab entries but never appear in
        normal text and produce garbage in decoded output.

        Returns (B, max_new) int64 generated token IDs.
        """
        self.eval()
        device = next(self.parameters()).device
        ctx    = prompt_ids.to(device)
        B      = ctx.size(0)

        # Build control-character suppression mask once
        suppress_mask: Optional[torch.Tensor] = None
        if suppress_control:
            bad_ids = list(range(0, 9)) + list(range(11, 32))  # keep tab(9) newline(10)
            suppress_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
            for bid in bad_ids:
                if bid < self.vocab_size:
                    suppress_mask[bid] = True

        generated   = torch.zeros(B, max_new, dtype=torch.long, device=device)
        done        = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_new):
            # Full forward pass: encoder (causal) → cells → language_head
            logits = self.forward(ctx, causal=True)                  # (B, vocab)
            logits = logits / max(temperature, 1e-8)

            # Suppress non-printable control-character byte tokens
            if suppress_mask is not None:
                logits = logits.masked_fill(suppress_mask.unsqueeze(0), -1e9)

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits = logits.masked_fill(logits < top_vals[:, -1:], -1e9)

            # Nucleus (top-p) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove   = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = -1e9
                logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

            if temperature < 1e-4:
                next_tok = logits.argmax(dim=-1)
            else:
                probs   = F.softmax(logits, dim=-1)
                row_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                probs   = probs / row_sum
                next_tok = torch.multinomial(probs, 1).squeeze(-1)

            generated[:, t] = next_tok
            done |= (next_tok == eos_id)
            if done.all():
                break

            ctx = torch.cat([ctx, next_tok.unsqueeze(-1)], dim=1)

        return generated

    # ------------------------------------------------------------------
    # DNA serialisation — full topology + evolutionary state
    # ------------------------------------------------------------------

    def save_dna(self, path: str) -> None:
        """
        Save the complete model "DNA" — everything needed to resurrect the
        exact evolved state on any hardware:

          weights + buffers   : via state_dict (all nn.Parameters and
                                registered buffers, including homeostasis
                                energy/integrity/excitement/pain)
          DAG topology        : each cell's input_sources list (not in state_dict)
          EA instincts        : lifetime_contrib EMA per cell (not in state_dict)
          freeze state        : frozen bool per cell (not in state_dict)
          cell shape metadata : in_features, out_features, layer_idx so we can
                                reconstruct cells with the right dimensions before
                                loading weights
          reservoir buffer    : sampled replay sequences (optional continuity)
          constructor config  : all __init__ hyperparameters needed to rebuild
                                the model architecture before loading weights

        Load with HydraLM.load_dna(path).
        """
        # Per-cell topology + EA state (not captured by state_dict)
        cell_dna = []
        for cell in self.cells:
            cell_dna.append({
                "in_features":      cell.in_features,
                "out_features":     cell.out_features,
                "layer_idx":        cell.layer_idx,
                "input_sources":    list(cell.input_sources),
                "frozen":           cell.frozen,
                "lifetime_contrib": getattr(cell, "lifetime_contrib", 0.0),
            })

        # Reservoir buffer sequences (CPU tensors → lists for JSON-safe storage)
        replay_seqs = [s.tolist() for s in self.replay_buf._buf]

        dna = {
            # Schema version for forward-compatibility
            "dna_version": 1,

            # Constructor hyperparameters — needed to rebuild the skeleton
            "config": {
                "vocab_size":           self.vocab_size,
                "pad_id":               self.pad_id,
                "encoder_dim":          self.encoder_dim,
                "max_hidden_dim":       self.max_hidden_dim,
                "min_width":            self.min_width,
                "max_width":            self.max_width,
                "n_dag_layers":         self.n_dag_layers,
                "health_decay":         self.health_decay,
                "freeze_age":           self.freeze_age,
                "freeze_health_thresh": self.freeze_health_thresh,
                "freeze_streak_needed": self.freeze_streak_needed,
                "growth_interval":      self.growth_interval,
                "min_growth_cells":     self.min_growth_cells,
                "plasticity_lr":        self.plasticity_lr,
                "replay_ratio":         self.replay_ratio,
                "max_cells":            self.max_cells,
                "w_curiosity":          self.w_curiosity,
                "w_exploration":        self.w_exploration,
                "w_consolidation":      self.w_consolidation,
                "max_loops":            self.max_loops,
                "exit_entropy_weight":  self.exit_entropy_weight,
                "loop_depth_weight":     self.loop_depth_weight,
                "reinforce_weight":      self.reinforce_weight,
                "max_connections":      self.max_connections,
                "_homeo_dim":           self._homeo_dim,
            },

            # Topology + EA instincts
            "cell_dna":   cell_dna,

            # Replay buffer
            "replay_seqs": replay_seqs,
            "replay_seen": self.replay_buf._seen,
        }

        # Save topology + config as JSON (human-readable, diff-able)
        json_path = path if path.endswith(".json") else path + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dna, f, indent=2)

        # Save weights + buffers as a companion .pt file
        pt_path = json_path.replace(".json", ".pt")
        torch.save(self.state_dict(), pt_path)

        # Save gene pool alongside DNA
        gp_path = json_path.replace(".json", "_genepool.pt")
        self.gene_pool.save(gp_path)

        n_cells  = len(self.cells)
        frozen   = sum(1 for c in self.cells if c.frozen)
        n_replay = len(replay_seqs)
        print(f"  [DNA] Saved to {json_path} + {pt_path}")
        print(f"        cells={n_cells} ({frozen} frozen) | "
              f"replay={n_replay} seqs | "
              f"gene_pool={len(self.gene_pool)} snapshots | "
              f"homeo: {self.homeostasis.summary()}")

    @classmethod
    def load_dna(
        cls,
        path:   str,
        device: str = "cpu",
        # These encoder params aren't stored in DNA (they live in the encoder
        # submodule's state_dict) but are needed to reconstruct the skeleton.
        # Pass them explicitly, or rely on defaults matching the original run.
        d_model:           int   = 256,
        n_heads:           int   = 4,
        n_layers:          int   = 4,
        max_seq_len:       int   = 512,
        dropout:           float = 0.1,
        gate_n_stages:     int   = 4,
        gate_n_probes:     int   = 5,
        gate_warmup_steps: int   = 500,
    ) -> "HydraLM":
        """
        Resurrect a model from saved DNA.

        Steps
        -----
        1. Load JSON topology + config.
        2. Reconstruct HydraLM with the exact same cell population shape
           (right number of cells, right in/out dimensions, right wiring).
        3. Load state_dict weights + buffers into the skeleton.
        4. Restore per-cell EA instincts (lifetime_contrib, frozen, input_sources).
        5. Restore replay buffer.

        The model is returned in eval mode on `device`.
        Call model.train() before resuming training.
        """
        json_path = path if path.endswith(".json") else path + ".json"
        pt_path   = json_path.replace(".json", ".pt")

        with open(json_path, "r", encoding="utf-8") as f:
            dna = json.load(f)

        cfg      = dna["config"]
        cell_dna = dna["cell_dna"]

        # Build the skeleton with enough initial_cells=0 so __init__ doesn't
        # create any cells — we'll add them manually to match the saved topology.
        model = cls(
            vocab_size           = cfg["vocab_size"],
            pad_id               = cfg["pad_id"],
            d_model              = d_model,
            n_heads              = n_heads,
            n_layers             = n_layers,
            max_seq_len          = max_seq_len,
            encoder_dim          = cfg["encoder_dim"],
            dropout              = dropout,
            gate_n_stages        = gate_n_stages,
            gate_n_probes        = gate_n_probes,
            gate_warmup_steps    = gate_warmup_steps,
            initial_cells        = 0,           # ← we rebuild topology manually
            max_cells            = cfg["max_cells"],
            min_width            = cfg.get("min_width", 16),
            max_width            = cfg.get("max_width", cfg.get("max_hidden_dim", 512)),
            n_dag_layers         = cfg["n_dag_layers"],
            health_decay         = cfg["health_decay"],
            freeze_age           = cfg["freeze_age"],
            freeze_health_thresh = cfg["freeze_health_thresh"],
            freeze_streak_needed = cfg["freeze_streak_needed"],
            growth_interval      = cfg["growth_interval"],
            min_growth_cells     = cfg["min_growth_cells"],
            plasticity_lr        = cfg["plasticity_lr"],
            replay_ratio         = cfg["replay_ratio"],
            max_loops            = cfg["max_loops"],
            exit_entropy_weight  = cfg["exit_entropy_weight"],
            loop_depth_weight    = cfg.get("loop_depth_weight", 0.01),
            reinforce_weight     = cfg.get("reinforce_weight", 0.01),
            max_connections      = cfg.get("max_connections", 10),
            w_curiosity          = cfg["w_curiosity"],
            w_exploration        = cfg["w_exploration"],
            w_consolidation      = cfg["w_consolidation"],
        )

        # Reconstruct cells in saved order with exact dimensions + wiring
        cells    = nn.ModuleList()
        adapters = nn.ModuleList()
        for cd in cell_dna:
            cell = HydraAdaptiveCell(
                in_features   = cd["in_features"],
                out_features  = cd["out_features"],
                plasticity_lr = cfg["plasticity_lr"],
                health_decay  = cfg["health_decay"],
                layer_idx     = cd["layer_idx"],
                input_sources = cd["input_sources"],
            )
            adapter = nn.Linear(cd["out_features"], cfg["max_hidden_dim"])
            cells.append(cell)
            adapters.append(adapter)

        model.cells         = cells
        model.cell_adapters = adapters

        # Load weights + buffers (fills in all nn.Parameters and registered
        # buffers including homeostasis scalars, health, age, etc.)
        state = torch.load(pt_path, map_location=device)
        model.load_state_dict(state, strict=True)

        # Restore Python-level EA instincts (not in state_dict)
        for i, (cell, cd) in enumerate(zip(model.cells, cell_dna)):
            cell.input_sources    = cd["input_sources"]   # re-apply after load
            cell.frozen           = cd["frozen"]
            cell.lifetime_contrib = cd["lifetime_contrib"]
            # Re-apply freeze so requires_grad matches
            if cd["frozen"]:
                cell.freeze()
            else:
                cell.unfreeze()

        # Restore replay buffer
        model.replay_buf._seen = dna.get("replay_seen", 0)
        model.replay_buf._buf  = [
            torch.tensor(seq, dtype=torch.long)
            for seq in dna.get("replay_seqs", [])
        ]

        # Restore gene pool (silently skips if file absent — starts fresh)
        gp_path = json_path.replace(".json", "_genepool.pt")
        model.gene_pool.load(gp_path)
        # Re-wire gene pool reference in rel_ea (rel_ea is not an nn.Module
        # so it isn't reconstructed by load_state_dict)
        model.rel_ea.gene_pool = model.gene_pool

        model = model.to(device)
        model.eval()

        n_cells = len(model.cells)
        frozen  = sum(1 for c in model.cells if c.frozen)
        print(f"  [DNA] Loaded from {json_path}")
        print(f"        cells={n_cells} ({frozen} frozen) | "
              f"gene_pool={len(model.gene_pool)} snapshots | "
              f"homeo: {model.homeostasis.summary()}")
        return model

    # ------------------------------------------------------------------
    # Optimizer surgery (after EA changes architecture)
    # ------------------------------------------------------------------

    def rebuild_optimizer(self, optimizer, lr: float):
        old_state = {p.data_ptr(): optimizer.state.get(p, {})
                     for group in optimizer.param_groups for p in group["params"]}
        trainable = [p for p in self.parameters() if p.requires_grad]
        new_opt   = torch.optim.Adam(trainable, lr=lr)
        for p in trainable:
            s = old_state.get(p.data_ptr(), {})
            if s:
                new_opt.state[p] = s
        return new_opt

    def summary(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen   = sum(1 for c in self.cells if c.frozen)
        return (f"HydraLM | vocab={self.vocab_size} | cells={len(self.cells)} "
                f"({frozen} frozen) | params={n_params:,} | "
                f"homeo: {self.homeostasis.summary()}")


# ===========================================================================
# 8.  Reservoir Buffer (minimal, no image-specific code)
# ===========================================================================

class _ReservoirBuffer:
    """Vitter Algorithm-R reservoir for text token sequences."""

    def __init__(self, capacity: int = 4000):
        self.capacity = capacity
        self._buf:  List[torch.Tensor] = []
        self._seen: int = 0

    def add_batch(self, seqs: torch.Tensor) -> None:
        """seqs: (B, L) int64."""
        for i in range(seqs.size(0)):
            self._seen += 1
            if len(self._buf) < self.capacity:
                self._buf.append(seqs[i].cpu())
            else:
                j = random.randint(0, self._seen - 1)
                if j < self.capacity:
                    self._buf[j] = seqs[i].cpu()

    def sample(self, n: int, device: str = "cpu") -> Optional[torch.Tensor]:
        if len(self._buf) < 2:
            return None
        items = random.choices(self._buf, k=min(n, len(self._buf)))
        # Pad to same length
        max_l = max(s.size(0) for s in items)
        padded = torch.stack([
            F.pad(s, (0, max_l - s.size(0)), value=SuperBPETokenizer.PAD_ID)
            for s in items
        ])
        return padded.to(device)

    def __len__(self):
        return len(self._buf)


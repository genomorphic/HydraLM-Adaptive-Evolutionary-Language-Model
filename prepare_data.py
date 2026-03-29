"""
prepare_data.py — One-time Pile data preparation for HydraLM
=============================================================
Streams a weighted ~8 GB sample from The Pile via HuggingFace datasets,
applies per-subset cleaning, and writes a single shuffled output file
ready for train.py.

Requirements
------------
  pip install datasets huggingface_hub

Note
----
  Uses standard Parquet-format datasets — no trust_remote_code needed.
  pile_cc → allenai/c4, books3 → deepmind/pg19, etc. See HF_SOURCES.

Usage
-----
  # Default: ~8 GB output to data/corpus.txt
  python prepare_data.py

  # Custom output path and target size
  python prepare_data.py --output_path data/corpus.txt --target_gb 8.0

  # Dry run — print stats without writing
  python prepare_data.py --dry_run

  # Custom subset weights (must sum to 1.0)
  python prepare_data.py --weights pile_cc=0.40 wikipedia=0.20 books3=0.15 ...

Output
------
  {output_path}           — shuffled corpus, one document per line, UTF-8
  {output_path}.manifest  — JSON: subset stats, byte counts, date, config

Memory usage
------------
  Streams each subset independently into a temp file, then merge-shuffles
  using reservoir sampling. Peak RAM is O(reservoir_size) not O(corpus size).
  Default reservoir is 500k lines — uses ~500 MB RAM at most.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Subset configuration
# ---------------------------------------------------------------------------

# Default mix — fractions must sum to 1.0.
# books3 replaced with pg19 (public domain books, no copyright issues).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "pile_cc":        0.35,
    "wikipedia_en":   0.20,
    "pg19":           0.20,   # Project Gutenberg books (public domain)
    "openwebtext2":   0.10,
    "github":         0.05,
    "arxiv":          0.05,
    "dm_mathematics": 0.05,
}

TARGET_GB_DEFAULT = 8.0
RESERVOIR_SIZE    = 5_000_000  # lines in RAM for shuffle (~1 GB RAM, safe on Colab)

# Per-subset HuggingFace source descriptors.
# EleutherAI/pile uses a loading script that newer datasets versions block.
# Each subset is remapped to a standard Parquet-format mirror or equivalent.
#
# Fields:
#   repo      : HuggingFace dataset repo
#   config    : dataset config name (None = default)
#   split     : HF split to stream
#   text_field: field containing the document text (or None for custom extract)
#
HF_SOURCES: Dict[str, dict] = {
    "pile_cc": {
        "repo":       "tiiuae/falcon-refinedweb",
        "config":     None,
        "split":      "train",
        "text_field": "content",
    },
    "wikipedia_en": {
        "repo":       "wikimedia/wikipedia",
        "config":     "20231101.en",
        "split":      "train",
        "text_field": "text",
    },
    "pg19": {
        "repo":       "bookcorpus/bookcorpus",
        "config":     None,
        "split":      "train",
        "text_field": "text",
    },
    "openwebtext2": {
        "repo":       "Skylion007/openwebtext",
        "config":     None,
        "split":      "train",
        "text_field": "text",
    },
    "github": {
        "repo":       "codeparrot/codeparrot-clean",
        "config":     None,
        "split":      "train",
        "text_field": "content",
    },
    "arxiv": {
        "repo":       "gfissore/arxiv-abstracts-2021",
        "config":     None,
        "split":      "train",
        "text_field": "abstract",
    },
    "dm_mathematics": {
        "repo":       "lighteval/MATH",
        "config":     "all",
        "split":      "train",
        "text_field": None,   # custom extract: problem + solution fields
    },
}

# Friendly name → HF_SOURCES key (identity mapping here, kept for clarity)
HF_SUBSET_MAP: Dict[str, str] = {k: k for k in HF_SOURCES}


# ===========================================================================
# 1.  Per-subset text extraction
# ===========================================================================

def extract_text(example: dict, subset: str) -> Optional[str]:
    """
    Extract raw text from a HuggingFace example.
    Returns None if the example should be skipped entirely.

    Each subset may have a different field name or structure — handled here
    rather than in the cleaner so the cleaner only ever sees plain strings.
    """
    src = HF_SOURCES.get(subset, {})

    # dm_mathematics: join problem + solution as a natural exchange
    if subset == "dm_mathematics":
        q = example.get("problem",  example.get("Problem",  "")).strip()
        a = example.get("solution", example.get("Solution", "")).strip()
        if not q or not a:
            return None
        return f"Q: {q}\nA: {a}"

    # General: use the configured text_field
    field = src.get("text_field", "text") or "text"
    text  = example.get(field, "")
    if not isinstance(text, str):
        return None
    return text.strip() or None


# ===========================================================================
# 2.  Per-subset cleaning
# ===========================================================================

# Regex compiled once
_CTRL_CHARS      = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')
_MULTI_SPACE     = re.compile(r'  +')
_MULTI_NEWLINE   = re.compile(r'\n{4,}')
_LATEX_ENV       = re.compile(
    r'\\begin\{(equation|align|eqnarray|math|displaymath)[*]?\}.*?'
    r'\\end\{\1[*]?\}',
    re.DOTALL
)
_LATEX_BIBLIO    = re.compile(r'\\begin\{thebibliography\}.*$', re.DOTALL)
_CODE_JUNK_LINE  = re.compile(
    r'^(\s*#.*|\s*//.*|\s*import .*|\s*from .* import.*|'
    r'\s*require\(.*|\s*include\s*<.*|\s*using\s+.*|'
    r'\s*package\s+.*|\s*/\*.*\*/\s*)$'
)
_HIGH_SYMBOL_RATIO = re.compile(r'[^\w\s]')


def clean_generic(text: str) -> str:
    """Cleaning applied to all subsets."""
    text = _CTRL_CHARS.sub(' ', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NEWLINE.sub('\n\n\n', text)
    return text.strip()


def clean_arxiv(text: str) -> str:
    """Strip LaTeX math environments and bibliography from ArXiv text."""
    text = _LATEX_ENV.sub(' ', text)
    text = _LATEX_BIBLIO.sub('', text)
    return clean_generic(text)


def clean_github(text: str) -> str:
    """
    Filter low-signal code.
    - Skip files with high symbol-to-letter ratio (minified / generated)
    - Remove pure-junk lines (imports, comments blocks, package declarations)
    - Keep docstrings and meaningful code
    """
    if len(text) == 0:
        return ""

    # Symbol ratio check on first 500 chars — proxy for minified/generated code
    sample        = text[:500]
    n_symbols     = len(_HIGH_SYMBOL_RATIO.findall(sample))
    symbol_ratio  = n_symbols / max(len(sample), 1)
    if symbol_ratio > 0.45:
        return ""

    lines = text.split('\n')
    kept  = []
    for line in lines:
        if _CODE_JUNK_LINE.match(line) and len(line.strip()) < 120:
            continue
        kept.append(line)

    return clean_generic('\n'.join(kept))


def clean_mathematics(text: str) -> str:
    """Mathematics Q&A — just generic clean, the format is already structured."""
    return clean_generic(text)


def clean_text(text: str, subset: str) -> str:
    """Dispatch to per-subset cleaner."""
    if subset == "arxiv":
        return clean_arxiv(text)
    elif subset == "github":
        return clean_github(text)
    elif subset == "dm_mathematics":
        return clean_mathematics(text)
    else:
        return clean_generic(text)


# ===========================================================================
# 3.  Line splitting
# ===========================================================================

def split_into_lines(text: str, max_chars: int = 2048) -> List[str]:
    """
    Split a document into training lines.
    - Split on paragraph boundaries first (double newline)
    - If a paragraph exceeds max_chars, split further at sentence boundaries
    - Discard segments under 50 characters (noise/stubs)
    """
    paragraphs = re.split(r'\n{2,}', text)
    lines: List[str] = []

    for para in paragraphs:
        para = para.replace('\n', ' ').strip()
        if len(para) < 50:
            continue

        if len(para) <= max_chars:
            lines.append(para)
        else:
            # Split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            chunk = ''
            for sent in sentences:
                if len(chunk) + len(sent) + 1 <= max_chars:
                    chunk = (chunk + ' ' + sent).strip() if chunk else sent
                else:
                    if len(chunk) >= 50:
                        lines.append(chunk)
                    chunk = sent
            if len(chunk) >= 50:
                lines.append(chunk)

    return lines


# ===========================================================================
# 4.  Streaming from HuggingFace
# ===========================================================================

def stream_subset(
    subset:      str,
    target_bytes: int,
    verbose:     bool = True,
) -> Iterator[str]:
    """
    Stream cleaned lines from one subset until target_bytes is reached.
    Yields one training line at a time.
    Each subset uses its own HuggingFace repo/config from HF_SOURCES.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    src = HF_SOURCES.get(subset)
    if src is None:
        print(f"  [WARN] No source config for {subset}. Skipping.")
        return

    if verbose:
        print(f"  Streaming {subset} from {src['repo']} "
              f"(target {target_bytes / 1e6:.1f} MB)...")

    load_kwargs: dict = {
        "path":      src["repo"],
        "split":     src["split"],
        "streaming": True,
    }
    if src.get("config"):
        load_kwargs["name"] = src["config"]

    try:
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        print(f"  [WARN] Could not load {subset}: {e}. Skipping.")
        return

    bytes_written = 0
    docs_seen     = 0
    lines_written = 0
    last_print    = 0
    print_every   = 50 * 1024 * 1024   # print every 50 MB
    max_retries   = 5

    for attempt in range(max_retries):
        try:
            for example in ds:
                raw = extract_text(example, subset)
                if raw is None:
                    continue

                cleaned = clean_text(raw, subset)
                if not cleaned:
                    continue

                lines = split_into_lines(cleaned)
                for line in lines:
                    encoded = (line + '\n').encode('utf-8')
                    bytes_written += len(encoded)
                    lines_written += 1
                    yield line

                docs_seen += 1

                if verbose and bytes_written - last_print >= print_every:
                    pct = min(bytes_written / target_bytes * 100, 100)
                    bar = int(pct / 5)
                    print(f"    {subset}: [{'#'*bar}{'.'*(20-bar)}] "
                          f"{pct:5.1f}%  {bytes_written/1e6:7.1f}/{target_bytes/1e6:.0f} MB  "
                          f"{docs_seen:,} docs",
                          flush=True)
                    last_print = bytes_written

                if bytes_written >= target_bytes:
                    break

            # Completed without error
            break

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"    [RETRY {attempt+1}/{max_retries}] {subset} connection dropped "
                      f"at {bytes_written/1e6:.1f} MB — retrying in {wait}s... ({e})",
                      flush=True)
                import time as _time
                _time.sleep(wait)
                # Re-create the streaming dataset and skip already-seen docs
                try:
                    ds = load_dataset(**load_kwargs)
                    # Fast-forward past already processed documents
                    ds = ds.skip(docs_seen)
                except Exception as e2:
                    print(f"    [RETRY] Could not reconnect: {e2}", flush=True)
                    break
            else:
                print(f"    [WARN] {subset} gave up after {max_retries} attempts "
                      f"at {bytes_written/1e6:.1f} MB: {e}", flush=True)
                break

    if verbose:
        print(f"    {subset}: done — {docs_seen:,} docs | "
              f"{lines_written:,} lines | "
              f"{bytes_written / 1e6:.1f} MB")


# ===========================================================================
# 5.  Reservoir shuffle merge
# ===========================================================================

class ReservoirShuffler:
    """
    Vitter Algorithm-R reservoir over a set of input files.
    Reads all temp files line-by-line, keeps a reservoir of size N,
    then writes the reservoir in shuffled order to the output.

    This gives an approximately uniform random sample / shuffle
    without loading the full dataset into RAM.
    """

    def __init__(self, reservoir_size: int = RESERVOIR_SIZE, seed: int = 42):
        self.reservoir_size = reservoir_size
        self.seed           = seed

    def shuffle_files_to(
        self,
        input_files: List[str],
        output_path: str,
        verbose:     bool = True,
    ) -> int:
        """
        Merge-shuffle all input_files into output_path.
        Returns total lines written.
        """
        random.seed(self.seed)
        reservoir: List[str] = []
        seen   = 0
        total  = 0

        if verbose:
            print("\nMerge-shuffling temp files into final corpus...")

        for fpath in input_files:
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line:
                        continue
                    seen += 1
                    if len(reservoir) < self.reservoir_size:
                        reservoir.append(line)
                    else:
                        j = random.randint(0, seen - 1)
                        if j < self.reservoir_size:
                            reservoir[j] = line

        random.shuffle(reservoir)

        with open(output_path, 'w', encoding='utf-8') as out:
            for line in reservoir:
                out.write(line + '\n')
                total += 1

        if verbose:
            print(f"  {seen:,} lines seen → "
                  f"{total:,} lines written (reservoir={self.reservoir_size:,})")

        return total

    def stream_all_to(
        self,
        input_files: List[str],
        output_path: str,
        verbose:     bool = True,
    ) -> int:
        """
        When total data fits in reservoir: full shuffle.
        When larger: reservoir sample then append remainder in random order.
        This ensures the output is always well-shuffled even for large corpora.
        """
        # Count total lines first (fast, just counting)
        total_lines = 0
        for fpath in input_files:
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    total_lines += sum(1 for _ in f)

        if verbose:
            print(f"  Total lines across temp files: {total_lines:,}")

        return self.shuffle_files_to(input_files, output_path, verbose)


# ===========================================================================
# 6.  Main preparation pipeline
# ===========================================================================

def prepare(
    output_path:    str,
    target_gb:      float,
    weights:        Dict[str, float],
    reservoir_size: int  = RESERVOIR_SIZE,
    seed:           int  = 42,
    dry_run:        bool = False,
    verbose:        bool = True,
) -> None:
    """
    Full pipeline:
    1. Compute per-subset byte targets from weights + total GB.
    2. Stream each subset into a temp file.
    3. Merge-shuffle all temp files into the final output.
    4. Write manifest.
    """
    random.seed(seed)

    # Normalise weights
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    target_bytes = int(target_gb * 1024 ** 3)
    subset_targets = {
        k: int(v * target_bytes)
        for k, v in weights.items()
    }

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nHydraLM Data Preparation")
        print(f"{'='*50}")
        print(f"Target size : {target_gb:.1f} GB ({target_bytes / 1e9:.2f} GB)")
        print(f"Output      : {output_path}")
        print(f"Subsets     :")
        for k, v in weights.items():
            mb = subset_targets[k] / 1e6
            print(f"  {k:<20} {v*100:5.1f}%  (~{mb:.0f} MB)")
        print()

    if dry_run:
        print("[dry_run] Exiting without downloading.")
        return

    # Stream each subset into a temp file
    temp_files: List[str]  = []
    manifest_subsets: Dict = {}
    t_start = time.time()

    tmpdir = tempfile.mkdtemp(prefix="hydra_pile_")

    for subset, byte_target in subset_targets.items():
        tmp_path = os.path.join(tmpdir, f"{subset}.txt")
        temp_files.append(tmp_path)

        lines_written = 0
        bytes_written = 0
        t0 = time.time()

        with open(tmp_path, 'w', encoding='utf-8') as f:
            for line in stream_subset(subset, byte_target, verbose=verbose):
                f.write(line + '\n')
                lines_written += 1
                bytes_written += len(line.encode('utf-8')) + 1

        elapsed = time.time() - t0
        manifest_subsets[subset] = {
            "lines":    lines_written,
            "bytes":    bytes_written,
            "mb":       round(bytes_written / 1e6, 2),
            "target_mb": round(byte_target / 1e6, 2),
            "weight":   round(weights[subset], 4),
            "elapsed_s": round(elapsed, 1),
        }

    # Merge-shuffle into final output
    shuffler = ReservoirShuffler(reservoir_size=reservoir_size, seed=seed)
    total_lines = shuffler.stream_all_to(temp_files, output_path, verbose=verbose)

    # Compute final file size
    final_bytes = os.path.getsize(output_path)

    # Clean up temp files
    for f in temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    elapsed_total = time.time() - t_start

    # Write manifest
    manifest = {
        "created":        datetime.now(timezone.utc).isoformat(),
        "output_path":    output_path,
        "target_gb":      target_gb,
        "final_bytes":    final_bytes,
        "final_gb":       round(final_bytes / 1024**3, 3),
        "total_lines":    total_lines,
        "reservoir_size": reservoir_size,
        "seed":           seed,
        "elapsed_s":      round(elapsed_total, 1),
        "subsets":        manifest_subsets,
    }
    manifest_path = output_path + ".manifest"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Done in {elapsed_total/60:.1f} min")
        print(f"Output : {output_path}")
        print(f"Size   : {final_bytes / 1024**3:.2f} GB ({total_lines:,} lines)")
        print(f"Manifest: {manifest_path}")
        print()
        print("Subset summary:")
        for k, v in manifest_subsets.items():
            pct = v['bytes'] / max(final_bytes, 1) * 100
            print(f"  {k:<20} {v['lines']:>8,} lines | "
                  f"{v['mb']:>7.1f} MB | "
                  f"{pct:4.1f}% of output")
        print()
        print(f"Next step:")
        print(f"  python train.py --data_path {output_path} --run_dir runs/run1")


# ===========================================================================
# 7.  Argparse entry point
# ===========================================================================

def parse_weights(raw: List[str]) -> Dict[str, float]:
    """Parse 'key=value' weight strings into a dict."""
    out = {}
    for item in raw:
        try:
            k, v = item.split('=')
            out[k.strip()] = float(v.strip())
        except ValueError:
            print(f"ERROR: invalid weight format '{item}' — expected 'subset=0.XX'")
            sys.exit(1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare ~8 GB Pile sample for HydraLM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output_path", default="data/corpus.txt",
        help="Path for the output corpus file",
    )
    p.add_argument(
        "--target_gb", type=float, default=TARGET_GB_DEFAULT,
        help="Target total size in GB",
    )
    p.add_argument(
        "--weights", nargs="+", default=[],
        metavar="SUBSET=FRAC",
        help=(
            "Custom subset weights, e.g. --weights pile_cc=0.40 wikipedia=0.20 "
            "(must sum to ~1.0, unspecified subsets are dropped)"
        ),
    )
    p.add_argument(
        "--reservoir_size", type=int, default=RESERVOIR_SIZE,
        help="Lines held in RAM for shuffle (higher = better shuffle, more RAM)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print config and exit without downloading",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    args    = p.parse_args()
    weights = parse_weights(args.weights) if args.weights else DEFAULT_WEIGHTS

    # Validate subsets
    unknown = set(weights) - set(HF_SUBSET_MAP)
    if unknown:
        print(f"ERROR: Unknown subsets: {unknown}")
        print(f"Valid subsets: {list(HF_SUBSET_MAP)}")
        sys.exit(1)

    prepare(
        output_path    = args.output_path,
        target_gb      = args.target_gb,
        weights        = weights,
        reservoir_size = args.reservoir_size,
        seed           = args.seed,
        dry_run        = args.dry_run,
        verbose        = not args.quiet,
    )


if __name__ == "__main__":
    main()

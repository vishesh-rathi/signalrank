"""Offline embedding precompute: candidates + JD probes -> .npy artifacts.

Built to run on a small (8 GB RAM) machine. Candidates are streamed and embedded in
fixed-size CHUNKS; each chunk's vectors are kept as fp16 and the torch device
cache is cleared between chunks, so peak memory tracks one chunk's working set
instead of growing across the whole 100K pool. The naive "encode everything at
once" path ballooned past 6 GB (the MPS caching allocator never released freed
memory) and thrashed the OS.

Run once before ranking (network needed for the first model download):

    uv run --group precompute python precompute.py                       # CPU (safe default)
    uv run --group precompute python precompute.py --device mps --batch-size 32

Ranking then needs only numpy; rank.py never imports torch.
"""

import argparse
import gc
import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ranker import config
from ranker.embeddings import encode, load_model
from ranker.util import build_narrative, iter_candidates

# Short texts gain nothing from tokenizer fork-parallelism, which also leaks
# semaphores at shutdown and adds memory pressure on small machines.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def release_device(device: str) -> None:
    """Free the torch caching allocator between chunks (MPS/CUDA grow without it)."""
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def iter_narrative_chunks(
    path: str, chunk_size: int, limit: int
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (candidate_ids, narratives) batches of up to ``chunk_size`` records.

    Streams the JSONL so neither the full candidate list nor every narrative is
    ever resident at once.
    """
    ids: list[str] = []
    narratives: list[str] = []
    for processed, candidate in enumerate(iter_candidates(path)):
        if limit and processed >= limit:
            break
        ids.append(candidate.get("candidate_id") or f"ROW_{processed}")
        narratives.append(build_narrative(candidate))
        if len(narratives) >= chunk_size:
            yield ids, narratives
            ids, narratives = [], []
    if narratives:
        yield ids, narratives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--limit", type=int, default=0, help="embed only the first N (0 = all)")
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for this OFFLINE encode (cpu/mps/cuda); the contest's "
        "CPU-only rule binds the ranking step, not precompute",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="encode batch size")
    parser.add_argument(
        "--chunk-size", type=int, default=2000, help="records embedded per memory-bounded chunk"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="truncate narratives to this many tokens (halves cost vs the 512 default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(device=args.device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    # Probe embeddings double as the dimension check and the first device warm-up.
    probe_matrix = encode(model, config.JD_PROBES, batch_size=args.batch_size).astype(np.float16)

    candidate_ids: list[str] = []
    chunk_matrices: list[np.ndarray] = []
    for ids, narratives in iter_narrative_chunks(args.candidates, args.chunk_size, args.limit):
        vectors = encode(model, narratives, batch_size=args.batch_size).astype(np.float16)
        chunk_matrices.append(vectors)
        candidate_ids.extend(ids)
        release_device(args.device)
        gc.collect()
        print(f"embedded {len(candidate_ids)} candidates", flush=True)

    candidate_matrix = (
        np.concatenate(chunk_matrices, axis=0)
        if chunk_matrices
        else np.empty((0, probe_matrix.shape[1]), dtype=np.float16)
    )

    np.save(out_dir / "cand_embeddings.npy", candidate_matrix)
    np.save(out_dir / "jd_probe_embeddings.npy", probe_matrix)
    (out_dir / "cand_ids.json").write_text(json.dumps(candidate_ids))
    print(
        f"saved {len(candidate_ids)} candidate embeddings "
        f"(dim={candidate_matrix.shape[1]}) and {len(probe_matrix)} JD probes -> {out_dir}/"
    )


if __name__ == "__main__":
    main()

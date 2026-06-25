"""Embedding utilities split across the precompute/rank boundary.

``cosine_topk`` is the only function the ranking path uses — pure numpy, fed
by ``.npy`` artifacts written offline. ``load_model``/``encode`` belong to the
precompute side and import sentence-transformers (and therefore torch) lazily,
so ``rank.py`` can run inside the contest's CPU-only, no-network sandbox with
nothing heavier than numpy installed.
"""

import numpy as np


def cosine_topk(candidate_vector: np.ndarray, probe_matrix: np.ndarray, k: int = 3) -> float:
    """Mean of the top-k cosine similarities between a candidate and the JD probes.

    Artifacts are stored fp16, so cast up to fp32 before normalizing. The
    result is clamped to >= 0: a probe pointing away from the candidate means
    "no semantic signal", and must not drag an otherwise-good fit negative.
    """
    candidate = candidate_vector.astype(np.float32)
    candidate /= np.linalg.norm(candidate) + 1e-9
    probes = probe_matrix.astype(np.float32)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True) + 1e-9
    similarities = probes @ candidate
    top_k = np.sort(similarities)[::-1][:k]
    return float(max(0.0, top_k.mean()))


def semantic_fit_scores(embeddings: np.ndarray, probe_matrix: np.ndarray, k: int = 3) -> np.ndarray:
    """Pool-normalized semantic JD-similarity for every candidate row, in [0, 1].

    The raw top-k cosine is a real recall signal but useless as an absolute
    feature: on this pool it occupies ~[0.60, 0.69] — 98% of candidates sit above
    0.6 and the whole usable spread is <0.09 of the nominal [0, 1]. Min-max would
    keep that lopsided shape; instead we map each candidate to its PERCENTILE RANK
    within the pool, so the feature becomes "more JD-similar than X% of the pool" —
    a uniform [0, 1] recall axis that finally differentiates candidates.

    Deterministic (stable argsort, pure numpy, no new artifacts) so reruns stay
    byte-identical. Vectorized: one matmul over the whole matrix, not per row.
    """
    candidates = embeddings.astype(np.float32)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-9
    probes = probe_matrix.astype(np.float32)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True) + 1e-9
    similarities = candidates @ probes.T  # (n_candidates, n_probes)
    k = min(k, similarities.shape[1])
    raw = np.sort(similarities, axis=1)[:, ::-1][:, :k].mean(axis=1)
    raw = np.clip(raw, 0.0, None)
    n = raw.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = raw.argsort(kind="stable")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return ranks / (n - 1)


def load_model(name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
    """Load the sentence-transformer used at precompute time (downloads on first use).

    ``device`` may be "mps"/"cuda" to speed up the OFFLINE precompute — the
    contest's CPU-only constraint applies to the ranking step, which never
    touches this function.
    """
    # Imported here, not at module level: torch must never load during ranking.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def encode(
    model, texts: list[str], batch_size: int = 64, show_progress_bar: bool = False
) -> np.ndarray:
    """Embed ``texts`` into a (len(texts), dim) float array.

    The modest default batch size and silent progress suit precompute's
    memory-bounded chunk loop, which prints its own per-chunk progress.
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

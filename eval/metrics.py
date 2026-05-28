"""Information-retrieval metrics for search-quality evaluation.

Pure Python, no heavy dependencies. Every metric takes:

- ``ranked_ids``: the ordered list of ``product_id`` returned by a search
  (best first), and
- ``qrels``: the ground-truth judgments for that query as a
  ``{product_id: gain}`` mapping, where gain follows the WANDS convention:
  ``Exact -> 2``, ``Partial -> 1``, and anything not judged is treated as ``0``.

Relevance gain conventions
--------------------------
* **Graded** (NDCG, used as-is): Exact=2, Partial=1.
* **Strict binary** (Recall/Precision/MRR/MAP, ``strict=True``): only Exact
  counts as relevant.
* **Lenient binary** (``strict=False``): Exact OR Partial counts as relevant.

Run ``python -m eval.metrics`` to execute the built-in self-test.
"""
from __future__ import annotations

import math

EXACT = 2
PARTIAL = 1

Qrels = dict[int, int]  # {product_id: gain}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _relevant_ids(qrels: Qrels, strict: bool) -> set[int]:
    """The set of product_ids considered relevant under the given definition."""
    threshold = EXACT if strict else PARTIAL
    return {pid for pid, gain in qrels.items() if gain >= threshold}


# --------------------------------------------------------------------------- #
# graded metric: NDCG
# --------------------------------------------------------------------------- #
def dcg(gains: list[int]) -> float:
    """Discounted cumulative gain of an ordered list of gains."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[int], qrels: Qrels, k: int) -> float | None:
    """Normalized DCG@k using graded gains (Exact=2, Partial=1).

    Returns ``None`` when the query has no positive judgments (IDCG == 0), so
    callers can exclude it from the mean rather than averaging in a zero.
    """
    ideal_gains = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal_gains)
    if idcg == 0:
        return None
    run_gains = [qrels.get(pid, 0) for pid in ranked_ids[:k]]
    return dcg(run_gains) / idcg


# --------------------------------------------------------------------------- #
# binary metrics
# --------------------------------------------------------------------------- #
def recall_at_k(ranked_ids: list[int], qrels: Qrels, k: int, *, strict: bool) -> float | None:
    """|relevant ∩ top-k| / |relevant|. ``None`` if no relevant docs exist."""
    relevant = _relevant_ids(qrels, strict)
    if not relevant:
        return None
    hits = sum(1 for pid in ranked_ids[:k] if pid in relevant)
    return hits / len(relevant)


def precision_at_k(ranked_ids: list[int], qrels: Qrels, k: int, *, strict: bool) -> float | None:
    """|relevant ∩ top-k| / k. ``None`` if no relevant docs exist for the query."""
    relevant = _relevant_ids(qrels, strict)
    if not relevant:
        return None
    hits = sum(1 for pid in ranked_ids[:k] if pid in relevant)
    return hits / k


def mrr(ranked_ids: list[int], qrels: Qrels, *, strict: bool) -> float | None:
    """Reciprocal rank of the first relevant result. ``None`` if no relevant docs."""
    relevant = _relevant_ids(qrels, strict)
    if not relevant:
        return None
    for i, pid in enumerate(ranked_ids):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision_at_k(
    ranked_ids: list[int], qrels: Qrels, k: int, *, strict: bool
) -> float | None:
    """Average precision@k (the per-query term of MAP). ``None`` if no relevant."""
    relevant = _relevant_ids(qrels, strict)
    if not relevant:
        return None
    hits = 0
    score = 0.0
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in relevant:
            hits += 1
            score += hits / (i + 1)
    denom = min(len(relevant), k)
    return score / denom if denom else 0.0


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def mean(values: list[float | None]) -> float | None:
    """Mean over the non-``None`` values (queries with no judgments excluded)."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # qrels: doc 10 = Exact(2), doc 20 = Partial(1), doc 30 = Exact(2)
    qrels: Qrels = {10: EXACT, 20: PARTIAL, 30: EXACT}

    # Ideal ranking -> NDCG == 1.0
    ideal = [10, 30, 20, 99]
    assert abs(ndcg_at_k(ideal, qrels, 10) - 1.0) < 1e-9, "ideal NDCG must be 1.0"

    # A relevant doc first -> MRR 1.0; relevant doc at rank 2 -> 0.5
    assert mrr([10, 1, 2], qrels, strict=True) == 1.0
    assert mrr([1, 10, 2], qrels, strict=True) == 0.5

    # Recall strict: relevant(Exact)={10,30}; top-2 [10,20] catches only 10 -> 0.5
    assert recall_at_k([10, 20, 5], qrels, 2, strict=True) == 0.5
    # Recall lenient: relevant={10,20,30}; top-2 [10,20] catches 2/3
    assert abs(recall_at_k([10, 20, 5], qrels, 2, strict=False) - 2 / 3) < 1e-9

    # Precision@2 strict for [10,20,...] -> 1 relevant / 2 = 0.5
    assert precision_at_k([10, 20, 5], qrels, 2, strict=True) == 0.5

    # No-judgment query -> None everywhere (excluded from means)
    assert ndcg_at_k([1, 2], {}, 10) is None
    assert recall_at_k([1, 2], {1: 0}, 10, strict=True) is None

    # AP@k: relevant strict = {10,30}; ranking [10, x, 30] -> (1/1 + 2/3)/2
    expected_ap = (1.0 + 2 / 3) / 2
    assert abs(average_precision_at_k([10, 5, 30], qrels, 10, strict=True) - expected_ap) < 1e-9

    # mean ignores None
    assert mean([1.0, None, 0.0]) == 0.5

    print("eval.metrics self-test: OK")


if __name__ == "__main__":
    _selftest()

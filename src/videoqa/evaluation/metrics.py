"""Scoring for ActivityNet-QA, faithful to the official script.

What "official" means here
--------------------------
The dataset's own ``evaluation/eval.py`` compares strings with ``==`` and
nothing else. There is no stemming, no synonym table, no partial credit.
It reports accuracy per question type and a "Free" aggregate over types
3-8. :func:`official_accuracy` reproduces that exactly, so our baseline
number means the same thing the dataset authors' number means.

Why a second, normalized score exists
-------------------------------------
Exact match was designed for models that emit a single word from a closed
vocabulary. A generative VLM answers "Yes." or "The girl has long hair",
which scores zero under ``==`` no matter how correct it is. Reporting only
the strict number would understate the model; reporting only a lenient
number would overstate it and is the kind of claim the brief calls
dishonest. So both are computed, from the same generations, and both are
written to the report.

The normalization is deliberately conservative and fully specified in
:func:`normalize_answer`: case, surrounding punctuation, and articles.
It never searches the generation for the right answer -- that would make
"not red" count as "red".

Raw generations are always saved. Any scoring rule we think of later can
be applied without re-running the GPU.
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Mirrors the official eval.py grouping: types 0-2 are reported
# individually, 3-8 are additionally pooled as "Free".
TYPE_NAMES: dict[int, str] = {
    0: "motion",
    1: "spatial_relation",
    2: "temporal_relation",
    3: "yes_no",
    4: "color",
    5: "object",
    6: "location",
    7: "number",
    8: "other",
}
FREE_TYPE_IDS = (3, 4, 5, 6, 7, 8)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace.

    This is the whole normalization: no synonyms, no substring search, no
    number-word conversion. Anything more generous would need its own
    justification in the report.
    """
    if text is None:
        return ""
    t = text.strip().lower()
    t = t.translate(_PUNCT_TABLE)
    t = _ARTICLES.sub(" ", t)
    return _WHITESPACE.sub(" ", t).strip()


@dataclass(slots=True)
class ScoreBreakdown:
    """Accuracy overall, per question type, and over the Free pool."""

    overall: float
    free: float
    per_type: dict[str, float]
    counts: dict[str, int]
    correct: dict[str, int]
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 6),
            "free": round(self.free, 6),
            "per_type": {k: round(v, 6) for k, v in self.per_type.items()},
            "counts": self.counts,
            "correct": self.correct,
            "n": self.n,
        }


def _score(records: Sequence[dict[str, Any]], matcher) -> ScoreBreakdown:
    correct: dict[int, int] = defaultdict(int)
    count: dict[int, int] = defaultdict(int)
    for r in records:
        tid = r.get("type_id")
        tid = tid if tid in TYPE_NAMES else 8
        count[tid] += 1
        if matcher(r):
            correct[tid] += 1

    n = sum(count.values())
    if n == 0:
        return ScoreBreakdown(0.0, 0.0, {}, {}, {}, 0)

    # .get() rather than indexing: a defaultdict would materialise
    # zero-count types here and pollute the reported breakdown.
    free_n = sum(count.get(t, 0) for t in FREE_TYPE_IDS)
    free_c = sum(correct.get(t, 0) for t in FREE_TYPE_IDS)
    return ScoreBreakdown(
        overall=sum(correct.values()) / n,
        free=(free_c / free_n) if free_n else 0.0,
        per_type={TYPE_NAMES[t]: correct.get(t, 0) / count[t] for t in sorted(count) if count[t]},
        counts={TYPE_NAMES[t]: count[t] for t in sorted(count)},
        correct={TYPE_NAMES[t]: correct.get(t, 0) for t in sorted(count)},
        n=n,
    )


def official_accuracy(records: Sequence[dict[str, Any]]) -> ScoreBreakdown:
    """Exact string equality, as the dataset's own eval.py does it."""
    return _score(records, lambda r: str(r.get("prediction", "")) == str(r.get("answer", "")))


def normalized_accuracy(records: Sequence[dict[str, Any]]) -> ScoreBreakdown:
    """Equality after :func:`normalize_answer` on both sides."""
    return _score(
        records,
        lambda r: normalize_answer(str(r.get("prediction", ""))) == normalize_answer(str(r.get("answer", ""))),
    )


def score_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Both scorings, plus the gap between them.

    A large gap means the model is right but badly formatted, which is a
    prompting problem, not a capability problem -- worth knowing before
    concluding anything about DoRA.
    """
    strict = official_accuracy(records)
    norm = normalized_accuracy(records)
    return {
        "official_exact_match": strict.to_dict(),
        "normalized": norm.to_dict(),
        "formatting_gap": round(norm.overall - strict.overall, 6),
    }


def categorize_errors(records: Iterable[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    """A small labelled sample of failures, as the brief's metrics require.

    Categories are assigned from evidence we actually have: whether the
    clip decoded, whether the answer was merely formatted differently,
    and whether the model refused to commit to a short answer. Anything
    needing human judgement is left as ``wrong_answer`` rather than
    guessed at.
    """
    out: list[dict[str, Any]] = []
    for r in records:
        pred, gold = str(r.get("prediction", "")), str(r.get("answer", ""))
        if normalize_answer(pred) == normalize_answer(gold):
            continue
        if r.get("decode_failed"):
            category = "decode_failure"
        elif not pred.strip():
            category = "empty_generation"
        elif len(pred.split()) > 8:
            category = "unconstrained_generation"
        elif gold in {"yes", "no"} and normalize_answer(pred) not in {"yes", "no"}:
            category = "unsupported_answer"
        else:
            category = "wrong_answer"
        out.append({
            "example_id": r.get("example_id"),
            "type_name": r.get("type_name"),
            "question": r.get("question"),
            "answer": gold,
            "prediction": pred,
            "category": category,
        })
        if len(out) >= limit:
            break
    return out

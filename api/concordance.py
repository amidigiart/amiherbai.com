# -*- coding: utf-8 -*-
"""
Concordance between two independent responses — anti-confabulation core.

Trigram-cosine similarity + hard numeric conflict check.
Agreement ≠ truth: two models can agree and still be wrong.
Fail direction is SAFE: false disagreement → honest "I don't know",
not false assertion.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip()


def _trigrams(text: str) -> dict[str, int]:
    t = _normalize(text)
    out: dict[str, int] = {}
    for i in range(len(t) - 2):
        g = t[i:i + 3]
        out[g] = out.get(g, 0) + 1
    return out


def trigram_cosine(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    common = set(ta) & set(tb)
    dot = sum(ta[g] * tb[g] for g in common)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    return dot / (na * nb)


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUM_RE.findall(text)}


@dataclass
class Concordance:
    score: float
    numeric_conflict: bool
    numbers_a: set = field(default_factory=set)
    numbers_b: set = field(default_factory=set)
    agree: bool = False
    reason: str = ""


def check_concordance(a: str, b: str, threshold: float = 0.45) -> Concordance:
    score = trigram_cosine(a, b)
    na, nb = numbers_in(a), numbers_in(b)
    a_only, b_only = na - nb, nb - na
    numeric_conflict = bool(a_only) and bool(b_only)

    if numeric_conflict and score < 0.70:
        return Concordance(score, True, na, nb, agree=False,
                           reason="contradictory numbers between engines")
    if score >= threshold:
        return Concordance(score, False, na, nb, agree=True,
                           reason=f"similarity {score:.2f} >= threshold {threshold}")
    return Concordance(score, False, na, nb, agree=False,
                       reason=f"similarity {score:.2f} < threshold {threshold}")

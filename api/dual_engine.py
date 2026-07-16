# -*- coding: utf-8 -*-
"""
AmiQiAI Dual Engine — Grok + DeepSeek anti-confabulation core.

Both models answer independently. If they agree (trigram cosine +
no numeric conflict), the response is delivered. If they disagree,
amiQiAI says so honestly instead of guessing. If one model is down,
the other's response is delivered with an explicit "uncorroborated" tag.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from adapters import AdapterError
from concordance import check_concordance


@dataclass
class DualResult:
    reply: str
    decision: str       # "affirm" | "disagree" | "degraded"
    concordance: float | None = None
    engine: str = ""    # which engine(s) produced this
    latency_s: float = 0.0


HONEST_DISAGREEMENT = (
    "I can't give you a confident answer here — my two internal sources "
    "don't agree, and rather than guess, I'd rather be honest. "
    "Can you rephrase, or shall we look at this from a different angle?"
)

UNCORROBORATED_TAG = (
    "\n\n(Note: this answer comes from a single engine and could not be "
    "cross-checked. Treat with appropriate caution.)"
)

BOTH_DOWN = (
    "I'm having a technical issue right now and don't want to improvise. "
    "Please try again in a moment."
)


class DualEngine:
    def __init__(self, adapter_a, adapter_b, system_prompt: str,
                 threshold: float = 0.45):
        self.a = adapter_a
        self.b = adapter_b
        self.system_prompt = system_prompt
        self.threshold = threshold

    def ask(self, user_msg: str) -> DualResult:
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(self.a.complete, self.system_prompt, user_msg)
            fb = ex.submit(self.b.complete, self.system_prompt, user_msg)
            ans_a = self._safe(fa)
            ans_b = self._safe(fb)

        lat = round(time.time() - t0, 2)

        if isinstance(ans_a, Exception) and isinstance(ans_b, Exception):
            return DualResult(reply=BOTH_DOWN, decision="degraded",
                              engine="none", latency_s=lat)

        if isinstance(ans_a, Exception):
            return DualResult(
                reply=str(ans_b) + UNCORROBORATED_TAG,
                decision="degraded", engine=self.b.name,
                latency_s=lat)

        if isinstance(ans_b, Exception):
            return DualResult(
                reply=str(ans_a) + UNCORROBORATED_TAG,
                decision="degraded", engine=self.a.name,
                latency_s=lat)

        conc = check_concordance(ans_a, ans_b, self.threshold)

        if conc.agree:
            return DualResult(
                reply=ans_a, decision="affirm",
                concordance=round(conc.score, 3),
                engine=f"{self.a.name}+{self.b.name}",
                latency_s=lat)

        return DualResult(
            reply=HONEST_DISAGREEMENT, decision="disagree",
            concordance=round(conc.score, 3),
            engine=f"{self.a.name}+{self.b.name}",
            latency_s=lat)

    @staticmethod
    def _safe(future):
        try:
            return future.result()
        except Exception as e:
            return e if isinstance(e, AdapterError) else AdapterError(str(e))

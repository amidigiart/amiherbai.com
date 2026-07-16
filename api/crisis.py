# -*- coding: utf-8 -*-
"""
Crisis detection — intercepts acute-risk messages BEFORE any model call.

When triggered, no API call is made. The response is a pre-written,
safety-reviewed message pointing to real helplines. This is not AI
judgment — it's pattern matching on known crisis language.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


_PATTERNS = [
    r"\b(vreau|voi|o)\s+(sa\s+)?(ma\s+)?(sinucid|omor|ucid)\b",
    r"\b(want|going)\s+to\s+(kill|end)\s+(myself|my life|it all)\b",
    r"\bsuicid(e|al|ar)?\b",
    r"\bself[- ]harm\b",
    r"\bnu (mai )?vreau sa traiesc\b",
    r"\bdon'?t want to (live|be alive|exist)\b",
    r"\bend (it|my life|everything)\b",
    r"\bma taie?\b.*\b(intentionat|singur)\b",
    r"\b(cut|hurt) myself\b",
    r"\bmi-e? (sa nu|ca o sa) (mor|dispar)\b",
]

_COMPILED = re.compile("|".join(_PATTERNS), re.IGNORECASE)

CRISIS_RESPONSES = {
    "en": (
        "I hear you, and what you're feeling matters. I'm not equipped to help "
        "with this — but real people are, right now.\n\n"
        "**Please reach out:**\n"
        "- Crisis Text Line: text HOME to 741741\n"
        "- National Suicide Prevention: 988 (US) or 116 123 (EU)\n"
        "- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n\n"
        "You don't have to go through this alone."
    ),
    "ro": (
        "Te aud, și ce simți contează. Nu sunt echipat să te ajut cu asta — "
        "dar oameni reali sunt, chiar acum.\n\n"
        "**Te rog, sună:**\n"
        "- Telefonul Sufletului: 0800 801 200 (gratuit, 24/7)\n"
        "- Linia pentru Viață: 0800 801 200\n"
        "- Ambulanța: 112\n\n"
        "Nu trebuie să treci prin asta singur/singură."
    ),
}


@dataclass
class CrisisSignal:
    is_crisis: bool
    matched_pattern: str = ""


def detect_crisis(text: str) -> CrisisSignal:
    t = _norm(text)
    m = _COMPILED.search(t)
    if m:
        return CrisisSignal(is_crisis=True, matched_pattern=m.group())
    return CrisisSignal(is_crisis=False)


def crisis_response(locale: str) -> str:
    return CRISIS_RESPONSES.get(locale, CRISIS_RESPONSES["en"])

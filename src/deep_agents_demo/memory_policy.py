"""Write policy for long-term memory: secrets are never stored.

``find_secrets`` names what looks secret in a text (never the value itself), and
``redact`` masks it. The memory tools refuse to store anything ``find_secrets``
flags, re-check what Mem0 actually stored after its own LLM rewrite, and redact
memory text before showing it, as a last line of defence for memories written
before this policy existed.

Detection is deliberately conservative: a false positive costs one refused
"remember", a false negative leaks a credential into a vector store.
"""
import math
import re
from collections import Counter

_PATTERNS = {
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
    "OpenAI or Anthropic API key": r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}",
    "NVIDIA API key": r"\bnvapi-[A-Za-z0-9_\-]{20,}",
    "AWS access key": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "GitHub token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
    "Google API key": r"\bAIza[0-9A-Za-z_\-]{35}\b",
    "Stripe key": r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}",
    "JSON web token": r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
    "credentials in a URL": r"\b[a-z][a-z0-9+.\-]*://[^\s/:@]+:[^\s/@]+@",
    "bearer token": r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{16,}",
    "password or secret value": (
        r"(?i)\b(?:password|passwd|passphrase|passcode|pwd|pin|secret|token|api[_ -]?key|"
        r"access[_ -]?key|client[_ -]?secret|private[_ -]?key)\b"
        r"\s*(?:is|was|=|:|->)\s*[\"']?[^\s\"',;]{4,}"
    ),
}
_COMPILED = {name: re.compile(rx) for name, rx in _PATTERNS.items()}
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_TOKEN = re.compile(r"[A-Za-z0-9+/_\-=]{32,}")


def _luhn(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def _random_tokens(text):
    """Long, mixed-character, high-entropy strings: probably keys, not words."""
    for m in _TOKEN.finditer(text):
        tok = m.group()
        classes = sum(bool(re.search(rx, tok)) for rx in (r"[a-z]", r"[A-Z]", r"\d"))
        if classes == 3 and _entropy(tok) >= 4.0:
            yield m


def _matches(text):
    for name, rx in _COMPILED.items():
        for m in rx.finditer(text):
            yield name, m
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            yield "payment card number", m
    for m in _random_tokens(text):
        yield "random-looking token", m


def find_secrets(text: str) -> list[str]:
    """Names of the kinds of secrets found in ``text`` (empty if none)."""
    return sorted({name for name, _ in _matches(text or "")})


def redact(text: str) -> str:
    """``text`` with every detected secret replaced by ``[REDACTED]``."""
    spans = sorted({m.span() for _, m in _matches(text or "")})
    out, last = [], 0
    for start, end in spans:
        if start < last:  # overlapping matches
            start = last
        if start < end:
            out.append(text[last:start])
            out.append("[REDACTED]")
            last = end
    out.append(text[last:])
    return "".join(out)

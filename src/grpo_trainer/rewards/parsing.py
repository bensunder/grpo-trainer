"""Pure text utilities shared by reward functions, the profiler and eval.

Nothing here raises on malformed model output; it returns None and the caller scores 0.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

# The full output contract: one think block, then one answer block, nothing else (whitespace allowed).
_CONTRACT_RE = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# A number token: optional sign, digits with optional thousands separators, optional decimals.
_NUMBER_RE = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")
_GSM8K_GT_RE = re.compile(r"####\s*(.+?)\s*\Z", re.DOTALL)


def completion_to_text(completion: Any) -> str:
    """Normalise a TRL completion to plain text.

    Standard format: `str`. Conversational format: `[{"role": "assistant", "content": ...}, ...]`,
    where content may itself be a string or a list of typed parts. Unknown shapes become "".
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _content_to_text(completion.get("content"))
    if isinstance(completion, (list, tuple)):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                if msg.get("role", "assistant") == "assistant":
                    parts.append(_content_to_text(msg.get("content")))
            elif isinstance(msg, str):
                parts.append(msg)
        return "".join(parts)
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    out.append(text)
        return "".join(out)
    return ""


def normalize_text(text: str) -> str:
    """NFKC folds full-width digits and compatibility characters; keeps the rest intact."""
    return unicodedata.normalize("NFKC", text)


def match_contract(text: str) -> tuple[str, str] | None:
    """Return (think, answer) if text is exactly `<think>..</think><answer>..</answer>`, else None.

    Rejects duplicate tags: each tag must occur exactly once.
    """
    text = normalize_text(text)
    for tag in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE):
        if text.count(tag) != 1:
            return None
    m = _CONTRACT_RE.match(text)
    if m is None:
        return None
    return m.group("think"), m.group("answer")


def extract_answer(text: str) -> str | None:
    """Content of the single <answer> block. None if absent or present more than once."""
    text = normalize_text(text)
    blocks = _ANSWER_RE.findall(text)
    if len(blocks) != 1:
        return None
    if text.count(ANSWER_OPEN) != 1 or text.count(ANSWER_CLOSE) != 1:
        return None  # unbalanced or nested tags
    return blocks[0]


def parse_single_number(text: str) -> Decimal | None:
    """Parse the one number in `text`.

    Accepts: "72", "$72", "72.00", "1,234", "-5", "72 apples", "**72**", "\\boxed{72}".
    Rejects: no number, or more than one *distinct* number (prevents answer spraying like "72 or 73").
    """
    if not text:
        return None
    t = normalize_text(text).replace("−", "-")  # unicode minus
    tokens = _NUMBER_RE.findall(t)
    values: set[Decimal] = set()
    for tok in tokens:
        v = _to_decimal(tok)
        if v is not None:
            values.add(v)
    if len(values) != 1:
        return None
    return next(iter(values))


def _to_decimal(token: str) -> Decimal | None:
    cleaned = token.replace(",", "").replace("−", "-").lstrip("+")
    try:
        d = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    return d.normalize() if d != 0 else Decimal(0)


def parse_gsm8k_ground_truth(answer_field: str) -> str | None:
    """GSM8K `answer` ends with `#### <number>`; return the normalised number as a string."""
    if not isinstance(answer_field, str):
        return None
    m = _GSM8K_GT_RE.search(answer_field.strip())
    if m is None:
        return None
    value = parse_single_number(m.group(1))
    return None if value is None else format_decimal(value)


def format_decimal(value: Decimal) -> str:
    """Canonical string: no exponent, no trailing zeros ('72', '0.5', '-3')."""
    s = format(value.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def numbers_equal(predicted: Decimal | None, ground_truth: str | None) -> bool:
    if predicted is None or ground_truth is None:
        return False
    gt = _to_decimal(ground_truth)
    return gt is not None and predicted == gt

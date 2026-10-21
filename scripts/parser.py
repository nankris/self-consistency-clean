"""Final-answer parser for GSM8K generations.

Robust extractor designed to handle standard conversational answers
without falling prey to placeholder token copying or Markdown styling.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)(?![0-9A-Za-z])")
_BOXED = re.compile(r"\\boxed\s*\{\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)(?![0-9A-Za-z])")
_HASH = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")

# Pattern matching variants of "The answer is ...", "Answer: ...", etc.
# Permissive to leading words like "The", "Final", markdown asterisks, colons, equals signs.
_ANSWER_PATTERNS = [
    re.compile(r"(?:final\s+)?answer\s*(?:is|=|:)\s*(?:the\s+)?[\$*`\s]*([-+]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(?:final\s+)?answer\s*:\s*(?:the\s+)?[\$*`\s]*([-+]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(?:the\s+answer\s+is|is\s+equal\s+to)\s*(?:the\s+)?[\$*`\s]*([-+]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
]


def canon(value) -> str | None:
    """Canonical fixed-point string representation for comparison.
    Normalizes '1,000', '1000', '1000.0', and '$1000' to '1000'.
    """
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if text == "":
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in ("", "-"):
        rendered = "0"
    return rendered


def _after_think(text: str) -> str:
    """Strip reasoning trace if thinking mode was present."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def parse_answer(text: str) -> str | None:
    """Extract canonical numerical answer from generation text.
    
    Tries in order:
    1. Explicit '#### <number>' (GSM8K native delimiter)
    2. LaTeX '\\boxed{<number>}'
    3. 'The answer is <number>', 'Answer: <number>', etc. (taking the last occurrence)
    4. Last line matching number at end of line.
    """
    if not text:
        return None
    
    body = _after_think(text).strip()
    
    # Check #### first (often produced by fine-tuned reasoning models)
    hash_matches = list(_HASH.finditer(body))
    if hash_matches:
        c = canon(hash_matches[-1].group(1))
        if c is not None:
            return c
            
    # Check \boxed{}
    boxed_matches = list(_BOXED.finditer(body))
    if boxed_matches:
        c = canon(boxed_matches[-1].group(1))
        if c is not None:
            return c

    # Check Answer patterns (search backwards from end of response)
    for pat in _ANSWER_PATTERNS:
        matches = list(pat.finditer(body))
        if matches:
            c = canon(matches[-1].group(1))
            if c is not None:
                return c
                
    # Fallback: scan lines from bottom to top for explicit answer indicator
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in reversed(lines):
        for pat in _ANSWER_PATTERNS:
            m = pat.search(line)
            if m:
                c = canon(m.group(1))
                if c is not None:
                    return c
        # If line starts with "Answer:" or "Therefore, the answer is..."
        if any(kw in line.lower() for kw in ["answer is", "answer:", "final answer"]):
            nums = _NUMBER.findall(line)
            if nums:
                c = canon(nums[-1])
                if c is not None:
                    return c

    return None


def vote(parsed_samples: list[str | None], k: int) -> str | None:
    """Plurality vote over the first k samples.
    
    Ties broken in favor of the earliest sample in the list.
    Unparsed samples (None) do not count toward valid votes and cannot win ties.
    Returns None if all k samples failed to parse.
    """
    window = parsed_samples[:k]
    counts: dict[str, int] = {}
    for val in window:
        if val is not None:
            counts[val] = counts.get(val, 0) + 1
            
    if not counts:
        return None
        
    max_count = max(counts.values())
    for val in window:
        if val is not None and counts[val] == max_count:
            return val
            
    return None

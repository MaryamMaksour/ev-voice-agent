"""Word error rate (WER) and key-term checks for speech-to-text, with text normalisation so that
"800,000", "800 thousand" and "eight hundred thousand" all count as the same words."""

from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"), (100, "hundred")]


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return f"{number_words(head)} {name}" + ("" if rest == 0 else " " + number_words(rest))
    return str(n)


def _number_token(tok: str) -> str:
    """'800,000' -> 'eight hundred thousand'; '1.2' -> 'one point two'; '4' -> 'four'."""
    tok = tok.replace(",", "")
    if re.fullmatch(r"\d+", tok):
        return number_words(int(tok)) if len(tok) <= 12 else " ".join(_ONES[int(d)] for d in tok)
    if re.fullmatch(r"\d+\.\d+", tok):
        whole, frac = tok.split(".")
        return number_words(int(whole)) + " point " + " ".join(_ONES[int(d)] for d in frac)
    return tok


def normalize(text: str) -> list[str]:
    text = (text or "").lower()
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)          # "4pm" -> "4 pm"
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)           # thousands separators
    text = re.sub(r"[^\w\s.']", " ", text, flags=re.UNICODE)  # keep letters, digits, decimal points, apostrophes
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)         # full stops, not decimal points
    words = []
    for tok in text.split():
        words.extend(_number_token(tok).split())
    # "800 thousand" and "eight hundred thousand" are now identical
    return [w.strip("'") for w in words if w.strip("'")]


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(substitutions + deletions + insertions, reference word count): Levenshtein over words."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1], len(ref)


def wer(reference: str, hypothesis: str) -> float:
    errors, n = word_errors(reference, hypothesis)
    return errors / n if n else float(bool(normalize(hypothesis)))


def key_terms_found(terms: list[str], hypothesis: str) -> tuple[list[str], list[str]]:
    """Which key terms (names, areas, amounts) appear in the transcript, word for word."""
    hyp = " " + " ".join(normalize(hypothesis)) + " "
    found, missed = [], []
    for term in terms:
        (found if " " + " ".join(normalize(term)) + " " in hyp else missed).append(term)
    return found, missed

"""Step 6: parse the final answer and score it (SQuAD/HotpotQA-style EM and F1)."""
import re
import string
from collections import Counter


def parse_final(text: str) -> str:
    """Take the last 'FINAL ANSWER: ...' line; fall back to the last non-empty line."""
    if not text:
        return ""
    matches = re.findall(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize(pred) == normalize(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    def one(p, g):
        p_tok, g_tok = normalize(p).split(), normalize(g).split()
        same = sum((Counter(p_tok) & Counter(g_tok)).values())
        if same == 0:
            return 0.0
        precision, recall = same / len(p_tok), same / len(g_tok)
        return 2 * precision * recall / (precision + recall)
    return max(one(pred, g) for g in golds)
"""Step 2: deterministic BM25 retrieval over one task's documents.

No model and no randomness: the same query always returns the same documents,
so retrieval stays constant across experimental conditions.
Reference: Robertson & Zaragoza (2009), The Probabilistic Relevance Framework: BM25 and Beyond.
"""
import math
import re
from collections import Counter

TOP_K = 3      # documents returned per search
K1 = 1.5       # term-frequency saturation (standard value)
B = 0.75       # length normalization strength (standard value)

STOPWORDS = frozenset("""
a an the of in on at to for and or but is are was were be been being by with as from
that this these those it its what which who whom whose when where why how did does do
has have had not no than then so such into about over also there their they he she his her
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words and numbers, drop stopwords."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25Index:
    """A BM25 search index over one task's documents."""

    def __init__(self, docs: list[str]):
        self.docs = docs
        self.term_counts = [Counter(tokenize(d)) for d in docs]
        self.doc_lengths = [sum(c.values()) for c in self.term_counts]
        self.avg_length = sum(self.doc_lengths) / len(docs)

        n = len(docs)
        doc_freq = Counter(word for counts in self.term_counts for word in counts)
        self.idf = {
            word: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for word, df in doc_freq.items()
        }

    def _score(self, query_terms: set[str], doc_id: int) -> float:
        counts = self.term_counts[doc_id]
        length_ratio = self.doc_lengths[doc_id] / self.avg_length
        score = 0.0
        for word in query_terms:
            tf = counts.get(word, 0)
            if tf == 0:
                continue
            score += self.idf[word] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length_ratio))
        return score

    def search(self, query: str, k: int = TOP_K) -> list[int]:
        """Return the ids of the k best-matching documents, best first."""
        query_terms = set(tokenize(query))
        scored = [(self._score(query_terms, i), i) for i in range(len(self.docs))]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for _, doc_id in scored[:k]]


def format_passages(docs: list[str], doc_ids: list[int]) -> str:
    """Turn retrieved documents into the text an agent receives."""
    return "\n\n".join(f"[Doc {i}] {docs[i]}" for i in doc_ids)


if __name__ == "__main__":
    from .data import load_tasks

    tasks = load_tasks()
    print("Retrieval quality, using each task's question as the query\n")
    for k in [1, 3, 5]:
        single = [t for t in tasks if t["complexity"] == "single_step"]
        multi = [t for t in tasks if t["complexity"] == "multi_step"]

        single_hit = sum(
            bool(set(BM25Index(t["docs"]).search(t["question"], k)) & set(t["gold_doc_ids"]))
            for t in single
        ) / len(single)

        multi_any, multi_all = 0, 0
        for t in multi:
            found = set(BM25Index(t["docs"]).search(t["question"], k))
            gold = set(t["gold_doc_ids"])
            multi_any += bool(found & gold)
            multi_all += gold <= found
        print(f"top-{k}:  single_step found gold = {single_hit:.0%}   |   "
              f"multi_step found any gold = {multi_any / len(multi):.0%}, "
              f"found ALL gold = {multi_all / len(multi):.0%}")
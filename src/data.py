"""Step 1: build the fixed benchmark task set.

single_step -> SQuAD v1.1: the answer is in one paragraph.
multi_step  -> HotpotQA (distractor): the answer needs facts from two paragraphs.

Each task carries its own small document collection (the gold paragraph(s) plus
distractors), so retrieval is real but cheap and fully reproducible.
"""
import json
import random
from collections import Counter

from datasets import load_dataset

from .config import N_DOCS_PER_TASK, N_PER_TYPE, SEED, TASKS_PATH


def _squad_tasks(rng: random.Random, n: int) -> list[dict]:
    squad = load_dataset("rajpurkar/squad", split="validation")
    all_contexts = sorted(set(squad["context"]))

    tasks = []
    for i in rng.sample(range(len(squad)), n):
        ex = squad[i]
        distractor_pool = [c for c in all_contexts if c != ex["context"]]
        docs = [ex["context"]] + rng.sample(distractor_pool, N_DOCS_PER_TASK - 1)
        rng.shuffle(docs)
        tasks.append({
            "task_id": f"squad-{ex['id']}",
            "complexity": "single_step",
            "source": "squad",
            "question": ex["question"],
            "answers": sorted(set(ex["answers"]["text"])),
            "docs": docs,
            "gold_doc_ids": [docs.index(ex["context"])],
        })
    return tasks


def _hotpot_tasks(rng: random.Random, n: int) -> list[dict]:
    hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    tasks = []
    for i in rng.sample(range(len(hotpot)), n):
        ex = hotpot[i]
        titles = ex["context"]["title"]
        docs = [
            f"{title}: " + " ".join(s.strip() for s in sentences)
            for title, sentences in zip(titles, ex["context"]["sentences"])
        ]
        gold_titles = set(ex["supporting_facts"]["title"])
        tasks.append({
            "task_id": f"hotpot-{ex['id']}",
            "complexity": "multi_step",
            "source": "hotpotqa",
            "question": ex["question"],
            "answers": [ex["answer"]],
            "docs": docs,
            "gold_doc_ids": [j for j, t in enumerate(titles) if t in gold_titles],
        })
    return tasks


def build_tasks() -> list[dict]:
    rng = random.Random(SEED)
    return _squad_tasks(rng, N_PER_TYPE) + _hotpot_tasks(rng, N_PER_TYPE)


def load_tasks() -> list[dict]:
    """Load the frozen task file, building it on first use. Delete the file to rebuild."""
    if not TASKS_PATH.exists():
        tasks = build_tasks()
        TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TASKS_PATH.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Built {len(tasks)} tasks -> {TASKS_PATH}")
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    tasks = load_tasks()
    print("Tasks per complexity:", dict(Counter(t["complexity"] for t in tasks)))
    for complexity in ["single_step", "multi_step"]:
        ex = next(t for t in tasks if t["complexity"] == complexity)
        print(f"\n--- example {complexity} ---")
        print("id:      ", ex["task_id"])
        print("question:", ex["question"])
        print("answers: ", ex["answers"])
        print("docs:    ", len(ex["docs"]), "| gold doc ids:", ex["gold_doc_ids"])
        print("gold doc:", ex["docs"][ex["gold_doc_ids"][0]][:200], "...")
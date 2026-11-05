"""Step 4 check: does short_summary compress messages, and does the key fact survive?"""
import asyncio
import re
import string

from src.data import load_tasks
from src.levels import Transformer, summary_target, word_count
from src.retrieval import format_passages

N_PER_TYPE = 10   # tasks per complexity to test


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text).strip()


def answer_in(text: str, answers: list[str]) -> bool:
    return any(normalize(a) in normalize(text) for a in answers)


async def main():
    tasks = load_tasks()
    full = Transformer("full")
    summary = Transformer("short_summary")

    for complexity in ("single_step", "multi_step"):
        selected = [t for t in tasks if t["complexity"] == complexity][:N_PER_TYPE]
        orig_words, summ_words = 0, 0
        checkable, kept_in_original, kept_in_summary = 0, 0, 0

        print("\n" + "=" * 70)
        print(complexity)
        for i, task in enumerate(selected):
            message = format_passages(task["docs"], task["gold_doc_ids"])

            assert await full(message) == message, "full level changed the message!"

            short = await summary(message)
            n, m = word_count(message), word_count(short)
            orig_words += n
            summ_words += m

            is_yes_no = normalize(task["answers"][0]) in ("yes", "no")
            if not is_yes_no:
                checkable += 1
                kept_in_original += answer_in(message, task["answers"])
                kept_in_summary += answer_in(short, task["answers"])

            survived = "n/a (yes/no)" if is_yes_no else answer_in(short, task["answers"])
            print(f"  {task['task_id']}: {n} -> {m} words "
                  f"(target {summary_target(n)}), answer kept: {survived}")

            if i == 0:
                print("\n  --- example ---")
                print("  Gold answer:", task["answers"])
                print("  ORIGINAL:", message[:400], "...")
                print("  SUMMARY: ", short, "\n")

        print(f"\n  Achieved compression: {summ_words / orig_words:.0%} of original words")
        if checkable:
            print(f"  Answer present in original: {kept_in_original}/{checkable}")
            print(f"  Answer present in summary:  {kept_in_summary}/{checkable}")


asyncio.run(main())
"""Step 6/7: run tasks x settings x levels, live or replay. Resumable."""
import argparse
import asyncio
import json
import time

from src.agents import BUILDERS
from src.config import MESSAGES_LOG, MODEL_DEPLOYMENT, RESULTS_LOG, RUN_OPTIONS, SUMMARY_DEPLOYMENT
from src.data import load_tasks
from src.intercept import MessageBus, load_frozen_messages
from src.levels import LEVELS, Transformer
from src.scoring import exact_match, f1, parse_final

SETTINGS = list(BUILDERS)


def finished_keys() -> set:
    if not RESULTS_LOG.exists():
        return set()
    with open(RESULTS_LOG, encoding="utf-8") as f:
        return {(r["task_id"], r["setting"], r["level"], r["mode"])
                for r in map(json.loads, f) if not r["error"]}


async def run_task(task, setting, level, mode, transformer, frozen_index):
    bus = MessageBus(level, transformer, MESSAGES_LOG, mode=mode)
    bus.start_task(task, setting, frozen=frozen_index.get((task["task_id"], setting)))
    system = BUILDERS[setting](task, bus)
    start, text, error = time.time(), "", None
    try:
        result = await system.entry.run(f"Question: {task['question']}", options=RUN_OPTIONS)
        text = result.text
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    found = {i for call in system.retrieval_calls for i in call["doc_ids"]}
    gold = set(task["gold_doc_ids"])
    prediction = parse_final(text)
    return {
        "task_id": task["task_id"], "complexity": task["complexity"],
        "setting": setting, "level": level, "mode": mode,
        "model": MODEL_DEPLOYMENT, "summary_model": SUMMARY_DEPLOYMENT,
        "question": task["question"], "gold": task["answers"],
        "prediction": prediction, "raw_output": text,
        "em": exact_match(prediction, task["answers"]),
        "f1": f1(prediction, task["answers"]),
        "n_searches": len(system.retrieval_calls),
        "all_gold_retrieved": gold <= found,
        "seconds": round(time.time() - start, 2),
        "error": error, **bus.stats(),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "replay"], default="live")
    ap.add_argument("--limit", type=int, default=None, help="tasks per complexity")
    ap.add_argument("--settings", nargs="+", choices=SETTINGS, default=SETTINGS)
    args = ap.parse_args()

    tasks = load_tasks()
    if args.limit:
        tasks = [t for c in ("single_step", "multi_step")
                 for t in [x for x in tasks if x["complexity"] == c][:args.limit]]

    levels = LEVELS if args.mode == "live" else ("short_summary",)
    frozen_index = load_frozen_messages(MESSAGES_LOG) if args.mode == "replay" else {}
    done = finished_keys()
    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)

    for level in levels:
        transformer = Transformer(level)
        for setting in args.settings:
            for i, task in enumerate(tasks, 1):
                if (task["task_id"], setting, level, args.mode) in done:
                    continue
                if args.mode == "replay" and (task["task_id"], setting) not in frozen_index:
                    print(f"  skip {task['task_id']} ({setting}): no full-context run to replay")
                    continue
                row = await run_task(task, setting, level, args.mode, transformer, frozen_index)
                with open(RESULTS_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                flag = f"  ERROR {row['error']}" if row["error"] else ""
                print(f"[{args.mode}|{level}|{setting}] {i}/{len(tasks)} {task['task_id']} "
                      f"EM={row['em']:.0f} F1={row['f1']:.2f} pred={row['prediction']!r}{flag}")


if __name__ == "__main__":
    asyncio.run(main())
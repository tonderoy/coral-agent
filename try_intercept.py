"""Step 5 check: run one task at full, summary (live) and summary (replay); inspect the log."""
import asyncio
import json

from src.agents import BUILDERS
from src.config import PROJECT_ROOT, RUN_OPTIONS
from src.data import load_tasks
from src.intercept import MessageBus, load_frozen_messages
from src.levels import Transformer

TEST_LOG = PROJECT_ROOT / "logs" / "test_messages.jsonl"
TASK_NUMBER = 0
SETTING = "multi_agent"


async def run_once(task, level, mode, frozen_index):
    bus = MessageBus(level, Transformer(level), TEST_LOG, mode=mode)
    bus.start_task(task, SETTING, frozen=frozen_index.get((task["task_id"], SETTING)))
    system = BUILDERS[SETTING](task, bus)
    result = await system.entry.run(f"Question: {task['question']}", options=RUN_OPTIONS)
    final = result.text.strip().splitlines()[-1] if result.text.strip() else "(no output)"
    print(f"\n[{level} | {mode}] final line: {final}")
    print(f"  stats: {bus.stats()}")
    return bus.run_id


def messages_of(run_id):
    with open(TEST_LOG, encoding="utf-8") as f:
        return [m for m in map(json.loads, f) if m["run_id"] == run_id]


def print_run(title, messages):
    print(f"\n--- {title} ---")
    for m in messages:
        flag = "  (replayed)" if m["replayed"] else ""
        print(f"  {m['step']:>2}  {m['sender']:>12} -> {m['receiver']:<12} "
              f"{m['words_original']:>4} -> {m['words_sent']:<4} words{flag}")


async def main():
    if TEST_LOG.exists():
        TEST_LOG.unlink()

    task = [t for t in load_tasks() if t["complexity"] == "multi_step"][TASK_NUMBER]
    print("Task:", task["task_id"])
    print("Question:", task["question"])
    print("Gold answer:", task["answers"])

    full_id = await run_once(task, "full", "live", {})
    summary_id = await run_once(task, "short_summary", "live", {})
    frozen_index = load_frozen_messages(TEST_LOG)
    replay_id = await run_once(task, "short_summary", "replay", frozen_index)

    full_msgs = messages_of(full_id)
    summary_msgs = messages_of(summary_id)
    replay_msgs = messages_of(replay_id)
    print_run("full | live", full_msgs)
    print_run("short_summary | live", summary_msgs)
    print_run("short_summary | replay", replay_msgs)

    print("\n--- automatic checks ---")
    assert all(m["sent"] == m["original"] for m in full_msgs), "full level changed a message"
    print("  OK  full level: every message delivered unchanged")

    assert any(m["words_sent"] < m["words_original"] for m in summary_msgs), "nothing was compressed"
    print("  OK  short_summary: at least one message was compressed")

    for original, replayed in zip(full_msgs, replay_msgs):
        if replayed["replayed"]:
            assert replayed["original"] == original["original"], "replayed text differs from log"
    n_replayed = sum(m["replayed"] for m in replay_msgs)
    print(f"  OK  replay: {n_replayed} of {len(replay_msgs)} messages reused the frozen originals")


asyncio.run(main())
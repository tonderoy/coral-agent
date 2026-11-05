"""Step 3 check: run both setups on two tasks and print every inter-agent message."""
import asyncio
import textwrap

from src.agents import BUILDERS, DirectChannel
from src.config import RUN_OPTIONS
from src.data import load_tasks

TASK_NUMBER = 0      # which task of each complexity to run (0 to 49)
PREVIEW_CHARS = 300  # how much of each message to print


class PrintChannel(DirectChannel):
    """A DirectChannel that also prints each message as it passes through."""

    async def pass_message(self, sender, receiver, content=None, produce=None):
        message = await super().pass_message(sender, receiver, content, produce)
        preview = message if len(message) <= PREVIEW_CHARS else message[:PREVIEW_CHARS] + " ..."
        print(f"\n  [{sender} -> {receiver}]")
        print(textwrap.indent(preview, "    "))
        return message


async def main():
    tasks = load_tasks()
    examples = [
        [t for t in tasks if t["complexity"] == complexity][TASK_NUMBER]
        for complexity in ("single_step", "multi_step")
    ]

    for task in examples:
        for setting, build in BUILDERS.items():
            print("\n" + "=" * 70)
            print(f"{setting} | {task['complexity']} | {task['task_id']}")
            print("Question:", task["question"])
            print("Gold answer:", task["answers"], "| gold doc ids:", task["gold_doc_ids"])

            system = build(task, PrintChannel())
            result = await system.entry.run(f"Question: {task['question']}", options=RUN_OPTIONS)

            print("\n  Searches made:")
            for call in system.retrieval_calls:
                print(f"    retrieve({call['query']!r}) -> {call['doc_ids']}")
            print("\n  Final output:")
            print(textwrap.indent(result.text, "    "))


asyncio.run(main())
"""Step 2 check: confirm an agent can call the retrieval tool on Azure."""
import asyncio
from typing import Annotated

from pydantic import Field
from agent_framework import Agent

from src.config import RUN_OPTIONS, make_client
from src.data import load_tasks
from src.retrieval import BM25Index, format_passages


async def main():
    task = next(t for t in load_tasks() if t["complexity"] == "multi_step")
    index = BM25Index(task["docs"])
    calls = []

    def retrieve(query: Annotated[str, Field(description="Keywords to search for")]) -> str:
        """Search the document collection and return the most relevant passages."""
        doc_ids = index.search(query)
        calls.append((query, doc_ids))
        return format_passages(task["docs"], doc_ids)

    agent = Agent(
        client=make_client(),
        name="ToolTest",
        instructions=(
            "Answer the question using the retrieve tool. You may search several times. "
            "End with one line: 'FINAL ANSWER: <short answer>'."
        ),
        tools=[retrieve],
    )
    result = await agent.run(f"Question: {task['question']}", options=RUN_OPTIONS)

    print("Question:     ", task["question"])
    print("Gold answer:  ", task["answers"])
    print("Gold doc ids: ", task["gold_doc_ids"])
    print("\nTool calls made by the agent:")
    for query, doc_ids in calls:
        print(f"  retrieve({query!r}) -> {doc_ids}")
    print("\nAgent output:\n" + result.text)


asyncio.run(main())
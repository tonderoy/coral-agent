"""Step 3: the single-agent and multi-agent setups.

single_agent: one agent with the retrieve tool.
multi_agent : Orchestrator --ask_retriever--> Retriever (has the retrieve tool)
                           --ask_solver-----> Solver
              Sub-agents are tools of the Orchestrator, so every delegation is a function
              call, and every message goes through a channel's pass_message() method.
"""
from dataclasses import dataclass, field
from typing import Annotated

from agent_framework import Agent
from pydantic import Field

from .config import RUN_OPTIONS, make_client
from .retrieval import BM25Index, format_passages

# ---------- instructions ----------
FINAL_ANSWER_RULE = (
    "End your reply with exactly one line in the form 'FINAL ANSWER: <answer>'. "
    "The answer must be as short as possible: a name, date, number, or yes/no. Not a sentence."
)

ORCHESTRATOR_INSTRUCTIONS = (
    "You lead a team that answers questions about a private document collection. "
    "You cannot see the documents; only the Retriever can. Work like this:\n"
    "1. Use ask_retriever to find the facts the question needs, one fact per request.\n"
    "2. If a fact you receive reveals something you must look up next (for example, the "
    "name of a person), ask the Retriever again about it.\n"
    "3. When you have enough evidence, call ask_solver once with the original question "
    "followed by all the evidence you gathered.\n"
    "4. Base your final answer on the Solver's reply. Do not answer from your own knowledge.\n"
    + FINAL_ANSWER_RULE
)

RETRIEVER_INSTRUCTIONS = (
    "You search a document collection for a teammate. Use the retrieve tool; if the first "
    "search misses, try different keywords. Reply with the facts that answer the request, "
    "quoting the key sentences. Report only what the documents say. If they do not contain "
    "the answer, say so."
)

SOLVER_INSTRUCTIONS = (
    "You answer a question using only the evidence you are given. Reply with only the "
    "answer: a name, date, number, or yes/no. If the evidence is incomplete, give your best "
    "answer from it."
)

SINGLE_AGENT_INSTRUCTIONS = (
    "You answer questions about a private document collection, which you can search with "
    "the retrieve tool. Search for each fact the question needs; if a result reveals "
    "something you must look up next (for example, a person's name), search again for it. "
    "Do not answer from your own knowledge.\n"
    + FINAL_ANSWER_RULE
)


# ---------- channel and delegation ----------
class DirectChannel:
    """Delivers every message unchanged. Stand-in for the MessageBus built in Step 5,
    which has the same pass_message() method but also summarizes and logs."""

    async def pass_message(self, sender: str, receiver: str,
                           content: str | None = None, produce=None) -> str:
        if content is None:
            content = await produce()
        return content


async def delegate(channel, sender: str, receiver: str, request: str, run_receiver) -> str:
    """Send a request, let the receiver work, send its reply back. Both messages go
    through the channel."""
    request_received = await channel.pass_message(sender, receiver, content=request)
    return await channel.pass_message(
        receiver, sender, produce=lambda: run_receiver(request_received)
    )


async def run_agent(agent: Agent, message: str) -> str:
    result = await agent.run(message, options=RUN_OPTIONS)
    return result.text


# ---------- building the setups ----------
@dataclass
class AgentSystem:
    entry: Agent                                         # the agent that receives the question
    retrieval_calls: list = field(default_factory=list)  # every search made, for analysis


def _make_retrieve_tool(task: dict, calls: list):
    index = BM25Index(task["docs"])

    def retrieve(query: Annotated[str, Field(description="Keywords to search for")]) -> str:
        """Search the document collection and return the most relevant passages."""
        doc_ids = index.search(query)
        calls.append({"query": query, "doc_ids": doc_ids})
        return format_passages(task["docs"], doc_ids)

    return retrieve


def build_single_agent(task: dict, channel) -> AgentSystem:
    calls: list = []
    search = _make_retrieve_tool(task, calls)

    async def retrieve(query: Annotated[str, Field(description="Keywords to search for")]) -> str:
        """Search the document collection and return the most relevant passages."""
        passages = search(query)
        return await channel.pass_message("RetrievalTool", "Agent", content=passages)

    agent = Agent(client=make_client(), name="Agent",
                  instructions=SINGLE_AGENT_INSTRUCTIONS, tools=[retrieve])
    return AgentSystem(entry=agent, retrieval_calls=calls)


def build_multi_agent(task: dict, channel) -> AgentSystem:
    calls: list = []
    retriever = Agent(client=make_client(), name="Retriever",
                      instructions=RETRIEVER_INSTRUCTIONS,
                      tools=[_make_retrieve_tool(task, calls)])
    solver = Agent(client=make_client(), name="Solver", instructions=SOLVER_INSTRUCTIONS)

    async def ask_retriever(
        request: Annotated[str, Field(
            description="One specific piece of information to find, "
                        "e.g. 'Who directed the film Inception?'")],
    ) -> str:
        """Ask the Retriever agent to find one piece of information in the documents.
        Returns its report."""
        return await delegate(channel, "Orchestrator", "Retriever", request,
                              lambda message: run_agent(retriever, message))

    async def ask_solver(
        request: Annotated[str, Field(
            description="The original question followed by all evidence gathered so far")],
    ) -> str:
        """Send the question and the gathered evidence to the Solver agent.
        Returns its short answer."""
        return await delegate(channel, "Orchestrator", "Solver", request,
                              lambda message: run_agent(solver, message))

    orchestrator = Agent(client=make_client(), name="Orchestrator",
                         instructions=ORCHESTRATOR_INSTRUCTIONS,
                         tools=[ask_retriever, ask_solver])
    return AgentSystem(entry=orchestrator, retrieval_calls=calls)


BUILDERS = {"single_agent": build_single_agent, "multi_agent": build_multi_agent}
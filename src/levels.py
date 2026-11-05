"""Step 4: the two communication levels.

full          -> the message is passed unchanged.
short_summary -> a separate Summarizer agent (fixed prompt, temperature 0) compresses the
                 message to SUMMARY_RATIO of its word count, never below MIN_SUMMARY_WORDS.
                 Messages already at or under the target pass unchanged. The summarizer does
                 not see the question (query-agnostic summary).

Report SUMMARY_RATIO, MIN_SUMMARY_WORDS, the summarizer deployment and
SUMMARIZER_INSTRUCTIONS in your methods: together they define the condition.
"""
from agent_framework import Agent

from .config import RUN_OPTIONS, SUMMARY_DEPLOYMENT, make_client

LEVELS = ("full", "short_summary")

SUMMARY_RATIO = 0.15      # target length = 15% of the original word count
MIN_SUMMARY_WORDS = 15    # ... but never shorter than this

SUMMARIZER_INSTRUCTIONS = (
    "You compress messages that AI agents send to each other. Rewrite the message as a "
    "short, faithful summary. Keep the names, numbers, dates and other specific facts it "
    "contains. Do not add information, do not answer any question in the message, and do "
    "not comment on it. Output only the summary."
)


def word_count(text: str) -> int:
    return len(text.split())


def summary_target(n_words: int) -> int:
    """How many words a summary of an n-word message may have."""
    return max(MIN_SUMMARY_WORDS, round(n_words * SUMMARY_RATIO))


class Transformer:
    """Applies one communication level to messages: await transformer(text) -> text."""

    def __init__(self, level: str):
        if level not in LEVELS:
            raise ValueError(f"Unknown level {level!r}; expected one of {LEVELS}")
        self.level = level
        self._cache: dict[str, str] = {}
        self._summarizer = None
        if level == "short_summary":
            self._summarizer = Agent(
                client=make_client(SUMMARY_DEPLOYMENT),
                name="Summarizer",
                instructions=SUMMARIZER_INSTRUCTIONS,
            )

    async def __call__(self, text: str) -> str:
        if self.level == "full":
            return text

        n_words = word_count(text)
        target = summary_target(n_words)
        if n_words <= target:
            return text

        if text not in self._cache:
            prompt = (
                f"Summarize the following message in at most {target} words.\n\n"
                f"MESSAGE:\n{text}"
            )
            result = await self._summarizer.run(prompt, options=RUN_OPTIONS)
            self._cache[text] = result.text.strip()
        return self._cache[text]
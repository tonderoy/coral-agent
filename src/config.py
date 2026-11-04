"""Shared configuration for the whole testbed.

Every setting that must be identical across experimental conditions lives here.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatClient

# ---------- project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

TASKS_PATH = PROJECT_ROOT / "data" / "tasks.json"
MESSAGES_LOG = PROJECT_ROOT / "logs" / "messages.jsonl"
RESULTS_LOG = PROJECT_ROOT / "logs" / "results.jsonl"

# ---------- Azure OpenAI connection ----------
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
AZURE_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
MODEL_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
SUMMARY_DEPLOYMENT = os.environ.get("AZURE_OPENAI_SUMMARY_DEPLOYMENT_NAME", MODEL_DEPLOYMENT)

# ---------- decoding ----------
RUN_OPTIONS = {"temperature": 0}

# ---------- task set ----------
N_PER_TYPE = 50        # tasks per complexity level (single_step, multi_step)
SEED = 42              # fixes which tasks are sampled
N_DOCS_PER_TASK = 10   # documents per task (gold + distractors)


def make_client(deployment: str | None = None) -> OpenAIChatClient:
    """Create a model client. All agents in the project are built through this function."""
    return OpenAIChatClient(
        base_url=f"{AZURE_ENDPOINT}/openai/v1/",
        api_key=AZURE_API_KEY,
        model=deployment or MODEL_DEPLOYMENT,
    )
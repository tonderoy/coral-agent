"""Step 0: confirm Agent Framework + Azure OpenAI access work."""
import asyncio
import os

from dotenv import load_dotenv
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

load_dotenv()


async def main():
    client = OpenAIChatClient(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    )
    agent = Agent(
        client=client,
        name="SmokeTest",
        instructions="Answer in one short sentence.",
    )
    result = await agent.run(
        "What is the capital of Australia?",
        options={"temperature": 0},
    )
    print("Agent:", result.text)


asyncio.run(main())
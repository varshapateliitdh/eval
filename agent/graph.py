"""
graph.py — LangGraph ReAct agent for financial Q&A.

Architecture
------------
The agent uses the ReAct pattern (Reason + Act):

  [user message]
       |
   [agent node]  ← LLM decides: answer now, or call a tool?
       |
  [tools node]   ← executes the chosen tool, returns result
       |
   [agent node]  ← LLM sees the tool result, decides next step
       ...
   [agent node]  ← LLM produces final answer
       |
    [END]

We use langgraph.prebuilt.create_react_agent which builds this loop for us.
All we need to provide is the LLM and the list of tools.
"""

import os
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.tools import ALL_TOOLS


def build_agent():
    """
    Build and return the LangGraph ReAct agent.

    Reads Azure OpenAI configuration from environment variables.
    Expected env vars:
        AZURE_OPENAI_API_KEY      — your Azure OpenAI key
        AZURE_OPENAI_ENDPOINT     — e.g. https://your-resource.openai.azure.com/
        AZURE_OPENAI_DEPLOYMENT   — your deployment name, e.g. gpt-4o or gpt-5
        AZURE_OPENAI_API_VERSION  — e.g. 2025-01-01-preview
    """
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        temperature=0,  # deterministic output — important for evals
    )

    system_prompt = (
        "You are a helpful financial assistant. "
        "You have access to tools to look up live stock prices and financial data. "
        "Always use the appropriate tool when asked about real market data. "
        "For calculations like ROI or percentage change, use the calculation tools. "
        "Be concise and precise in your answers."
    )

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=system_prompt,
    )

    return agent


def run_agent(agent, user_input: str) -> dict:
    """
    Run the agent on a single user input and return a structured result.

    Returns
    -------
    {
        "output":     str,        # the agent's final text answer
        "tool_calls": list[str],  # names of tools that were called
        "messages":   list,       # full message trace for debugging
    }
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]})

    messages = result["messages"]

    # Extract the final AI response (last AIMessage with content)
    output = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and msg.__class__.__name__ == "AIMessage":
            output = msg.content
            break

    # Collect all tool names that were called during the run
    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(tc["name"])

    return {
        "output": output,
        "tool_calls": tool_calls,
        "messages": messages,
    }

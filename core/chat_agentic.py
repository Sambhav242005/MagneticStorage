"""
Agentic chat implementation using Ollama function calling.
"""

import requests
import time


def chat_agentic(agent, user_input: str, MEMORY_TOOLS: list, base_url: str = "http://localhost:11434") -> str:
    """
    Agentic chat where the LLM decides when to search memory.

    Args:
        agent: NeuroSavant instance.
        user_input: User's message.
        MEMORY_TOOLS: Tool definitions for Ollama.

    Returns:
        The LLM response.
    """
    total_start = time.perf_counter()
    print(f"Thinking... (Model: {agent.config.model_name}) [AGENTIC MODE]")

    messages = [{"role": "user", "content": user_input}]
    full_reply = ""
    max_tool_calls = 5
    tool_calls_made = 0

    try:
        while tool_calls_made < max_tool_calls:
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": agent.config.model_name,
                    "messages": messages,
                    "tools": MEMORY_TOOLS,
                    "stream": False,
                },
            )

            if response.status_code != 200:
                print(f"\nWARN: Ollama API error: {response.text}")
                return "Error connecting to Ollama."

            result = response.json()
            message = result.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    tool_result = agent._execute_tool_call(tool_name, tool_args)

                    messages.append(message)
                    messages.append({"role": "tool", "content": tool_result})
                    tool_calls_made += 1
            else:
                full_reply = message.get("content", "")
                break

        if tool_calls_made >= max_tool_calls:
            print("\nWARN: Max tool calls reached")
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": agent.config.model_name,
                    "messages": messages,
                    "stream": False,
                },
            )
            full_reply = response.json().get("message", {}).get("content", "")

        print(f"Assistant: {full_reply}")

    except Exception as e:
        print(f"\nWARN: Generation error: {e}")
        full_reply = "I apologize, but I encountered an error."

    total_time = (time.perf_counter() - total_start) * 1000
    print(f"\nTiming: {total_time:.0f}ms total | Tool calls: {tool_calls_made}")

    return full_reply

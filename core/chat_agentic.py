"""
Agentic chat implementation using Ollama function calling
"""

import json
import requests
import time


def chat_agentic(agent, user_input: str, MEMORY_TOOLS: list) -> str:
    """
    Agentic chat where LLM decides when to search memory.
    
    Args:
        agent: NeuroSavant instance
        user_input: User's message
        MEMORY_TOOLS: Tool definitions for Ollama
    
    Returns:
        LLM's response
    """
    total_start = time.perf_counter()
    print(f"Thinking... (Model: {agent.config.model_name}) [AGENTIC MODE]")
    
    # Build conversation messages
    messages = [{"role": "user", "content": user_input}]
    full_reply = ""
    max_tool_calls = 5
    tool_calls_made = 0
    
    try:
        while tool_calls_made < max_tool_calls:
            # Call LLM with tools available
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": agent.config.model_name,
                    "messages": messages,
                    "tools": MEMORY_TOOLS,
                    "stream": False  # Non-streaming for tool calling
                }
            )
            
            if response.status_code != 200:
                print(f"\\n⚠️  Ollama API error: {response.text}")
                return "Error connecting to Ollama."
            
            result = response.json()
            message = result.get('message', {})
            
            # Check if LLM called a tool
            tool_calls = message.get('tool_calls', [])
            
            if tool_calls:
                # Execute tool call
                for tool_call in tool_calls:
                    tool_name = tool_call['function']['name']
                    tool_args = tool_call['function']['arguments']
                    
                    # Execute tool
                    tool_result = agent._execute_tool_call(tool_name, tool_args)
                    
                    # Add tool result to conversation
                    messages.append(message)  # LLM's tool call message
                    messages.append({
                        "role": "tool",
                        "content": tool_result
                    })
                    
                    tool_calls_made += 1
            else:
                # No tool call - LLM generated final response
                full_reply = message.get('content', '')
                break
        
        # If we hit max tool calls, get final response without tools
        if tool_calls_made >= max_tool_calls:
            print("\\n⚠️  Max tool calls reached")
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": agent.config.model_name,
                    "messages": messages,
                    "stream": False
                }
            )
            full_reply = response.json().get('message', {}).get('content', '')
        
        print(f"🤖 Assistant: {full_reply}")
        
    except Exception as e:
        print(f"\\n⚠️  Generation error: {e}")
        full_reply = "I apologize, but I encountered an error."
    
    total_time = (time.perf_counter() - total_start) * 1000
    print(f"\\n⏱️  Total: {total_time:.0f}ms | Tool calls: {tool_calls_made}")
    
    return full_reply

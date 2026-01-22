import json
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI


class Messages:
    """
    Manages conversation history for the agent.
    """

    def __init__(self, system_prompt: str):
        """
        Initialize the messages with a system prompt.
        
        Args:
            system_prompt: The system prompt to guide the assistant's behavior
        """
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, tool_calls: Optional[List] = None) -> None:
        """
        Add an assistant message to the conversation.
        
        Args:
            content: The text content of the message
            tool_calls: Optional list of tool calls made by the assistant
        """
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ]
        self._messages.append(message)

    def add_tool_message(self, result: Any, tool_call_id: str) -> None:
        """
        Add a tool response message to the conversation.
        
        Args:
            result: The result from the tool execution
            tool_call_id: The ID of the tool call this is responding to
        """
        self._messages.append({
            "role": "tool",
            "content": json.dumps(result) if isinstance(result, dict) else str(result),
            "tool_call_id": tool_call_id
        })

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return the full message history."""
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)


class LLM:
    """
    Wrapper for OpenAI-compatible LLM API interactions.
    """

    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        model: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096
    ):
        """
        Initialize the LLM client.
        
        Args:
            base_url: The API endpoint URL
            api_key: The API key for authentication
            model: The model identifier to use
            temperature: Sampling temperature (default 0.6)
            top_p: Top-p sampling parameter (default 0.95)
            max_tokens: Maximum tokens in response (default 4096)
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def query(
        self, 
        messages: Messages, 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Optional[List]]:
        """
        Send a query to the LLM and get a response.
        
        Args:
            messages: The conversation history
            tools: Optional list of tool schemas
            
        Returns:
            Tuple of (response content, tool calls if any)
        """
        kwargs = {
            "model": self.model,
            "messages": messages.get_messages(),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else None
        
        return content, tool_calls

    def strip_thinking(self, response: str) -> str:
        """
        Remove thinking tags from the response.
        
        Args:
            response: The raw response from the model
            
        Returns:
            The response with thinking content removed
        """
        if "</think>" in response:
            return response.split("</think>")[-1].strip()
        return response
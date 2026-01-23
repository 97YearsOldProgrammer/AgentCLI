import json as js
import openai as oai


#################### Default Model Configuration ####################

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL    = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"


#################### Inference Backend Options ####################

BACKENDS = {
    "vllm_local": {
        "base_url": "http://localhost:8000/v1",
        "model":    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "api_key":  "not-needed",
    },
    "ollama_local": {
        "base_url": "http://localhost:11434/v1",
        "model":    "nemotron-mini",
        "api_key":  "ollama",
    },
    "nvidia_cloud": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model":    "nvidia/nvidia-nemotron-nano-9b-v2",
        "api_key":  None,  # Requires env var
    },
}


#################### Messages Class ####################

class Messages:
    """

    Manages conversation history for the agent
    """

    def __init__(self, system_prompt):
        self._messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, content):
        """

        Add a user message to the conversation
        """
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content, tool_calls=None):
        """

        Add an assistant message to the conversation
        """
        message = {"role": "assistant", "content": content}

        if tool_calls:
            message["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ]

        self._messages.append(message)

    def add_tool_message(self, result, tool_call_id):
        """

        Add a tool response message to the conversation
        """
        content = js.dumps(result) if isinstance(result, dict) else str(result)
        self._messages.append({
            "role":         "tool",
            "content":      content,
            "tool_call_id": tool_call_id
        })

    def get_messages(self):
        """

        Return the full message history
        """
        return self._messages

    def __len__(self):
        return len(self._messages)


#################### LLM Class ####################

class LLM:
    """

    Wrapper for OpenAI-compatible LLM API interactions
    """

    def __init__(self, base_url, api_key, model, temperature=0.6, top_p=0.95, max_tokens=4096):
        self.client      = oai.OpenAI(base_url=base_url, api_key=api_key)
        self.model       = model
        self.temperature = temperature
        self.top_p       = top_p
        self.max_tokens  = max_tokens

    def query(self, messages, tools=None):
        """

        Send a query to the LLM and get a response
        """
        kwargs = {
            "model":       self.model,
            "messages":    messages.get_messages(),
            "temperature": self.temperature,
            "top_p":       self.top_p,
            "max_tokens":  self.max_tokens,
        }

        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = "auto"

        response   = self.client.chat.completions.create(**kwargs)
        message    = response.choices[0].message
        content    = message.content or ""
        tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else None

        return content, tool_calls

    def strip_thinking(self, response):
        """

        Remove thinking tags from the response
        """
        if "</think>" in response:
            return response.split("</think>")[-1].strip()
        return response
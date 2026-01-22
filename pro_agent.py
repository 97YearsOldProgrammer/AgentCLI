#!/usr/bin/env python3


import os
from pathlib    import Path
from typing     import Dict

from langgraph.prebuilt             import create_react_agent
from langgraph.checkpoint.memory    import InMemorySaver
from langchain_openai               import ChatOpenAI

from bash_tool import Bash, LIST_OF_ALLOWED_COMMANDS


# System prompt that guides the agent's behavior
SYSTEM_PROMPT = f"""/think
You are a helpful Bash assistant with the ability to execute commands in the shell.
You engage with users to help answer questions about bash commands, or execute their intent.
If user intent is unclear, keep engaging with them to figure out what they need and how to best help
them. If they ask question that are not relevant to bash or computer use, decline to answer.

When a command is executed, you will be given the output from that command and any errors. Based on
that, either take further actions or yield control to the user.

The bash interpreter's output and current working directory will be given to you every time a
command is executed. Take that into account for the next conversation.
If there was an error during execution, tell the user what that error was exactly.

You are only allowed to execute the following commands:
{', '.join(LIST_OF_ALLOWED_COMMANDS)}

**Never** attempt to execute a command not in this list. **Never** attempt to execute dangerous commands
like `rm`, `mv`, `rmdir`, `sudo`, etc. If the user asks you to do so, politely refuse.

When you switch to new directories, always list files so you can get more context.
"""


class ExecOnConfirm:
    """
    A wrapper around the Bash class to implement human-in-the-loop confirmation.
    This ensures the user approves each command before execution.
    """

    def __init__(self, bash: Bash):
        """
        Initialize with a Bash tool instance.
        
        Args:
            bash: The Bash tool instance to wrap
        """
        self.bash = bash

    def _confirm_execution(self, cmd: str) -> bool:
        """
        Ask the user whether the suggested command should be executed.
        
        Args:
            cmd: The command to be executed
            
        Returns:
            True if user confirms, False otherwise
        """
        return input(f"    ▶️   Execute '{cmd}'? [y/N]: ").strip().lower() == "y"

    def exec_bash_command(self, cmd: str) -> Dict[str, str]:
        """
        Execute a bash command after confirming with the user.
        
        Args:
            cmd: The bash command to execute
            
        Returns:
            Dictionary with execution results or error message
        """
        if self._confirm_execution(cmd):
            return self.bash.exec_bash_command(cmd)
        return {"error": "The user declined the execution of this command."}


def get_prompt_prefix(cwd: str) -> str:
    """
    Generate a prompt prefix showing the current working directory.
    
    Args:
        cwd: Current working directory
        
    Returns:
        Formatted prompt prefix string
    """
    return f"['{cwd}' 🙂] "


def strip_thinking(response: str) -> str:
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


def main():
    """Main entry point for the LangGraph-based bash agent."""
    
    # Configuration - can be overridden with environment variables
    base_url    = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key     = os.environ.get("LLM_API_KEY", "")
    model        = os.environ.get("LLM_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
    
    if not api_key:
        print("⚠️  Warning: LLM_API_KEY environment variable is not set.")
        print("   Please set it to your NVIDIA API key or OpenRouter API key.")
        print("   Get a free API key at: https://build.nvidia.com")
        print()
        api_key = input("Enter your API key (or press Enter to exit): ").strip()
        if not api_key:
            print("Exiting...")
            return
    
    # Get the starting directory
    start_dir = str(Path.home())
    
    # Instantiate the Bash class
    bash = Bash(cwd=start_dir, allowed_commands=LIST_OF_ALLOWED_COMMANDS)
    
    # Create the LangGraph agent
    agent = create_react_agent(
        model=ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
        ),
        tools=[ExecOnConfirm(bash).exec_bash_command],  # Wrap for human-in-the-loop
        prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    
    # Configuration for the agent's thread
    config = {"configurable": {"thread_id": "bash-agent-session"}}
    
    print("=" * 60)
    print("🖥️  NVIDIA Nemotron Bash Computer Use Agent (LangGraph)")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Starting directory: {start_dir}")
    print(f"Allowed commands: {len(LIST_OF_ALLOWED_COMMANDS)}")
    print()
    print("Type your instructions in natural language.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    print()
    
    # Create the user/agent interaction loop
    while True:
        try:
            # Get user input
            user_input = input(get_prompt_prefix(bash.cwd)).strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Run the agent's logic and get the response
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )
            
            # Show the response (without the thinking part, if any)
            response = result["messages"][-1].content.strip()
            response = strip_thinking(response)
            
            if response:
                print(f"\n[🤖] {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again.\n")


if __name__ == "__main__":
    main()
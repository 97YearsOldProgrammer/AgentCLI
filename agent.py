#!/usr/bin/env python3

import os
import json
from pathlib import Path

from bash_tool import Bash, LIST_OF_ALLOWED_COMMANDS
from llm_helpers import Messages, LLM


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


def confirm_execution(cmd: str) -> bool:
    """
    Ask the user whether the suggested command should be executed.
    
    Args:
        cmd: The command to be executed
        
    Returns:
        True if user confirms, False otherwise
    """
    response = input(f"    ▶️   Execute '{cmd}'? [y/N]: ").strip().lower()
    return response == "y"


def get_prompt_prefix(cwd: str) -> str:
    """
    Generate a prompt prefix showing the current working directory.
    
    Args:
        cwd: Current working directory
        
    Returns:
        Formatted prompt prefix string
    """
    return f"['{cwd}' 🙂] "


def main():
    """Main entry point for the bash agent."""
    
    # Configuration - can be overridden with environment variables
    base_url    = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key     = os.environ.get("LLM_API_KEY", "")
    model       = os.environ.get("LLM_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
    
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
    
    # Initialize components
    bash        = Bash(cwd=start_dir, allowed_commands=LIST_OF_ALLOWED_COMMANDS)
    llm         = LLM(base_url=base_url, api_key=api_key, model=model)
    messages    = Messages(SYSTEM_PROMPT)
    
    print("=" * 60)
    print("🖥️  NVIDIA Nemotron Bash Computer Use Agent")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Starting directory: {start_dir}")
    print(f"Allowed commands: {len(LIST_OF_ALLOWED_COMMANDS)}")
    print()
    print("Type your instructions in natural language.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    print()
    
    # The main agent loop
    while True:
        try:
            # Get user message
            user_input = input(get_prompt_prefix(bash.cwd)).strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            messages.add_user_message(user_input)
            
            # The tool-call/response loop
            while True:
                response, tool_calls = llm.query(messages, [bash.to_json_schema()])
                
                # Add the response to the context (with tool calls if any)
                messages.add_assistant_message(response, tool_calls)
                
                # Process tool calls
                if tool_calls:
                    for tc in tool_calls:
                        function_name = tc.function.name
                        try:
                            function_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            tool_call_result = {"error": "Failed to parse function arguments"}
                            messages.add_tool_message(tool_call_result, tc.id)
                            continue
                        
                        # Ensure it's calling the right tool
                        if function_name != "exec_bash_command" or "cmd" not in function_args:
                            tool_call_result = {"error": "Incorrect tool or function argument"}
                        else:
                            cmd = function_args["cmd"]
                            if confirm_execution(cmd):
                                tool_call_result = bash.exec_bash_command(cmd)
                            else:
                                tool_call_result = {"error": "The user declined the execution of this command."}
                        
                        messages.add_tool_message(tool_call_result, tc.id)
                else:
                    # No tool calls - display the assistant's response
                    clean_response = llm.strip_thinking(response)
                    if clean_response:
                        print(f"\n[🤖] {clean_response}")
                    print()
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again.\n")


if __name__ == "__main__":
    main()
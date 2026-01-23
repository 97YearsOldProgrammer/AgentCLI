#!/usr/bin/env python3

import os
import pathlib as pl

import langgraph.prebuilt as lgp
import langgraph.checkpoint.memory as lgm
import langchain_openai as lco

import lib.bash_tool as bt
import lib.llm_helpers as lh


#################### System Prompt ####################

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
{', '.join(bt.LIST_OF_ALLOWED_COMMANDS)}

**Never** attempt to execute a command not in this list. **Never** attempt to execute dangerous commands
like `rm`, `mv`, `rmdir`, `sudo`, etc. If the user asks you to do so, politely refuse.

When you switch to new directories, always list files so you can get more context.
"""


#################### Execution Wrapper Class ####################

class ExecOnConfirm:
    """

    A wrapper around Bash class to implement human-in-the-loop confirmation
    """

    def __init__(self, bash):
        self.bash = bash

    def _confirm_execution(self, cmd):
        """

        Ask the user whether the suggested command should be executed
        """
        if self.bash.is_auto_executable(cmd):
            print(f"    ⚡  Auto-executing '{cmd}'")
            return True

        return input(f"    ▶️   Execute '{cmd}'? [y/N]: ").strip().lower() == "y"

    def exec_bash_command(self, cmd):
        """

        Execute a bash command after confirming with the user
        """
        if self._confirm_execution(cmd):
            return self.bash.exec_bash_command(cmd)
        return {"error": "The user declined the execution of this command"}


#################### Helper Functions ####################

def get_prompt_prefix(cwd):
    """

    Generate a prompt prefix showing the current working directory
    """
    return f"['{cwd}' 🙂] "


def strip_thinking(response):
    """

    Remove thinking tags from the response
    """
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response


def get_api_key():
    """

    Prompt user for API key if not set in environment
    """
    api_key = os.environ.get("LLM_API_KEY", "")

    if not api_key:
        print("⚠️  Warning: LLM_API_KEY environment variable is not set")
        print("   Please set it to your NVIDIA API key or OpenRouter API key")
        print("   Get a free API key at: https://build.nvidia.com")
        print()
        api_key = input("Enter your API key (or press Enter to exit): ").strip()

    return api_key


def print_banner(model, start_dir, auto_count, allowed_count):
    """

    Print the startup banner with configuration info
    """
    print("=" * 60)
    print("🖥️  NVIDIA Nemotron Bash Computer Use Agent (LangGraph)")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Starting directory: {start_dir}")
    print(f"Allowed commands: {allowed_count}")
    print(f"Auto-execute commands: {auto_count}")
    print()
    print("Type your instructions in natural language")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 60)
    print()


#################### Main Entry Point ####################

def main():
    """

    Main entry point for the LangGraph-based bash agent
    """
    base_url = os.environ.get("LLM_BASE_URL", lh.DEFAULT_BASE_URL)
    api_key  = get_api_key()
    model    = os.environ.get("LLM_MODEL", lh.DEFAULT_MODEL)

    if not api_key:
        print("Exiting...")
        return

    start_dir = str(pl.Path.home())

    bash = bt.Bash(
        cwd=start_dir,
        allowed_commands=bt.LIST_OF_ALLOWED_COMMANDS,
        auto_execute_commands=bt.LIST_OF_AUTO_EXECUTE_COMMANDS
    )

    agent = lgp.create_react_agent(
        model=lco.ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
        ),
        tools=[ExecOnConfirm(bash).exec_bash_command],
        prompt=SYSTEM_PROMPT,
        checkpointer=lgm.InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "bash-agent-session"}}

    print_banner(
        model,
        start_dir,
        len(bt.LIST_OF_AUTO_EXECUTE_COMMANDS),
        len(bt.LIST_OF_ALLOWED_COMMANDS)
    )

    while True:
        try:
            user_input = input(get_prompt_prefix(bash.cwd)).strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            result   = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )

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
            print("   Please try again\n")


if __name__ == "__main__":
    main()